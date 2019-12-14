#!/usr/bin/env python

import torch
import time
import argparse
import numpy as np
import scipy.linalg

from blp95model import BLP95Model


def termination_check(
        x,
        residual,
        atol,
        rtol,
        norm):
    '''adapted from pyblp'''
    tol = atol
    if rtol > 0:
        tol += rtol * norm(x)
    res = norm(residual)
    return res < tol


def infinity_norm(x):
    '''adapted from pyblp'''
    return x.abs().max()


def invert_shares_squarem(
        initial,
        contraction,
        max_evaluations=500,
        atol=1e-12,
        rtol=1e-12,
        norm=infinity_norm,
        scheme=3,
        step_min=1.0,
        step_max=1.0,
        step_factor=4.0):
    '''adapted from pyblp'''
    x = initial
    failed = False
    evaluations = 0

    t_step_max = torch.tensor([step_max] * initial.shape[0], device=initial.device, dtype=t_type)
    t_step_min = torch.tensor([step_min] * initial.shape[0], device=initial.device, dtype=t_type)

    while True:
        # first step
        x0, x = x, contraction(x)
        if not torch.isfinite(x).all():
            x = x0
            failed = True
            break

        # check for convergence
        g0 = x - x0
        evaluations += 1
        if evaluations >= max_evaluations or termination_check(x, g0, atol, rtol, norm):
            break

        # second step
        x1, x = x, contraction(x)
        if not torch.isfinite(x).all():
            x = x1
            failed = True
            break

        # check for convergence
        g1 = x - x1
        evaluations += 1
        if evaluations >= max_evaluations or termination_check(x, g1, atol, rtol, norm):
            break

        # compute the step length
        r = g0
        v = g1 - g0

        if scheme == 1:
            # (r.T @ v) / (v.T @ v)
            alpha = torch.bmm(r[:, None, :], v[:, :, None]) / torch.bmm(v[:, None, :], v[:, :, None])
        elif scheme == 2:
            # (r.T @ r) / (r.T @ v)
            alpha = torch.bmm(r[:, None, :], r[:, :, None]) / torch.bmm(r[:, None, :], v[:, :, None])
        else:
            # -np.sqrt((r.T @ r) / (v.T @ v))
            alpha = -torch.sqrt(torch.bmm(r[:, None, :], r[:, :, None]) / torch.bmm(v[:, None, :], v[:, :, None]))

        alpha = alpha.squeeze(2)

        # bound the step length and update its bounds
        alpha = -torch.max(t_step_min[:, None], torch.min(t_step_max[:, None], -alpha))

        alpha_cond = (-alpha.squeeze() == t_step_max).float()
        t_step_max = alpha_cond * (t_step_max * step_factor) + (1 - alpha_cond) * t_step_max

        alpha_cond = ((-alpha.squeeze() == t_step_min) * (t_step_min < 0)).float()
        t_step_min = alpha_cond * (t_step_min * step_factor) + (1 - alpha_cond) * t_step_min

        # acceleration step
        x2, x = x, x0 - 2 * alpha * r + alpha**2 * v
        x3, x = x, contraction(x)
        if not torch.isfinite(x).all():
            x = x2
            failed = True
            break

        # check for convergence
        evaluations += 1
        if evaluations >= max_evaluations or termination_check(x, x - x3, atol, rtol, norm):
            break

    # determine whether there was convergence
    converged = not failed and evaluations < max_evaluations
    return x, converged


def batch_shares(
        delta,
        prod_char,
        market_masks,
        mdraws,
        mweights,
        theta,
        logshares=False,
        full_output=False):

    mu = torch.bmm(prod_char, mdraws * theta.view(-1, 1))
    util = mu + delta[:, :, None]
    util_max, __ = util.max(dim=1)
    exp_util = torch.exp(util - util_max[:, None, :]) * market_masks[:, :, None]
    total_exp_util = torch.exp(-util_max) + exp_util.sum(dim=1)
    indiv_shares = exp_util / total_exp_util[:, None, :]
    weighted_shares = indiv_shares * mweights
    s = weighted_shares.sum(dim=2)
    if logshares:
        s = torch.log(s + torch.abs(market_masks - 1))

    if full_output:
        return s, indiv_shares, weighted_shares
    else:
        return s


def batch_invert_shares(
        log_market_shares,
        prod_char,
        market_masks,
        mdraws,
        mweights,
        theta,
        delta_0=None,
        atol=1e-06,
        rtol=1e-06):

    if delta_0 is None:
        delta_0 = log_market_shares.new_zeros(log_market_shares.shape)

    def contraction(x):
        ls = batch_shares(x, prod_char, market_masks, mdraws, mweights, theta, logshares=True)
        return x + log_market_shares - ls
    delta_vec, res = invert_shares_squarem(delta_0, contraction, atol=atol, rtol=rtol)
    if not res:
        raise Exception('Failed to converge')

    return delta_vec, res


def gmm_loss(
        log_market_shares,
        prod_char,
        market_masks,
        own_mat,
        model_ids_to_rows,
        beta_mat,
        xw_mat,
        z_mat,
        mdraws,
        mweights,
        theta,
        gmm_weights,
        delta_0=None,
        full_output=False,
        atol=1e-06,
        rtol=1e-06):

    delta, __ = batch_invert_shares(
        log_market_shares,
        prod_char,
        market_masks,
        mdraws,
        mweights,
        theta,
        delta_0=delta_0,
        atol=atol,
        rtol=rtol)

    s, cs, ws = batch_shares(
        delta,
        prod_char,
        market_masks,
        mdraws,
        mweights,
        theta,
        full_output=True)

    ww = ws * mdraws[:, 0:1, :] * theta[0]
    Jp = torch.diag_embed(ww.sum(dim=2)) - torch.bmm(ww, torch.transpose(cs, 1, 2))
    OJp = Jp * own_mat + torch.diag_embed(1 - market_masks)
    eta, __ = torch.solve(-s[:, :, None], OJp)
    eta = eta.squeeze()
    costs = prod_char[:, :, 0] - eta

    mc = torch.masked_select(costs, market_masks.bool())
    mc[(mc < 0).detach()] = 0.001
    log_mc = torch.log(mc)

    y = torch.cat((
        torch.masked_select(delta, market_masks.bool()), log_mc), 0)
    beta = beta_mat @ y
    res = y - xw_mat @ beta

    g_hat = z_mat * res[:, None]
    g_hat_agg = model_ids_to_rows @ g_hat
    g_hat_mean = g_hat_agg.mean(dim=0)

    loss = g_hat_mean.t() @ gmm_weights @ g_hat_mean

    if full_output:
        with torch.no_grad():
            fact = 1.0 / (g_hat_agg.size(0) - 1)
            cov_mat = g_hat_agg - g_hat_mean
            cov_mat = fact * cov_mat.t().matmul(cov_mat)
            cov_mat = torch.inverse(cov_mat)
        return loss, delta, beta, cov_mat
    else:
        return loss, delta, beta


if __name__ == '__main__':
    t_tot = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-filename',
        type=str,
        required=True,
        help='data filename')

    parser.add_argument(
        '--unobs-filename',
        type=str,
        required=True,
        help='unobservables filename')

    parser.add_argument(
        '--dtype',
        type=str,
        default='float64',
        help='float32 or float64')

    parser.add_argument(
        '--disable-cuda',
        action='store_true',
        help='Disable CUDA')

    args = parser.parse_args()

    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    t_type = torch.float32
    rtol = atol = 1e-06
    if args.dtype == 'float64':
        t_type = torch.float64
        rtol = atol = 1e-12

    blp95 = BLP95Model(args.data_filename, args.unobs_filename)

    market_masks = torch.as_tensor(blp95.market_masks, device=device, dtype=t_type)
    mdraws = torch.as_tensor(blp95.draws, device=device, dtype=t_type)
    mweights = torch.as_tensor(blp95.weights, device=device, dtype=t_type)

    zx_mat = blp95.data[blp95.xi_moment_cols].values
    zw_mat = blp95.data[blp95.omega_moment_cols].values
    z_mat = scipy.linalg.block_diag(zx_mat, zw_mat)

    x_mat = blp95.data[blp95.delta_regressor_cols].values
    w_mat = blp95.data[blp95.cost_regressor_cols].values
    xw_mat = scipy.linalg.block_diag(x_mat, w_mat)

    weights_0 = np.linalg.inv(z_mat.T @ z_mat)

    proj_z = z_mat @ weights_0 @ z_mat.T
    xw_t_xw_inv = np.linalg.inv(xw_mat.T @ proj_z @ xw_mat)
    beta_mat = xw_t_xw_inv @ xw_mat.T @ proj_z

    z_mat = torch.as_tensor(z_mat, device=device, dtype=t_type)
    xw_mat = torch.as_tensor(xw_mat, device=device, dtype=t_type)
    proj_z = torch.as_tensor(proj_z, device=device, dtype=t_type)
    weights_0 = torch.as_tensor(weights_0, device=device, dtype=t_type)
    beta_mat = torch.as_tensor(beta_mat, device=device, dtype=t_type)

    prod_char = np.zeros((blp95.num_markets, blp95.max_market_size, blp95.num_prod_char))
    market_shares = np.zeros((blp95.num_markets, blp95.max_market_size))
    log_market_shares = np.zeros((blp95.num_markets, blp95.max_market_size))
    for i, mkt_id in enumerate(blp95.market_names):
        mx = blp95.data.loc[blp95.data.year == mkt_id, blp95.random_component_names].values
        prod_char[i, 0:mx.shape[0], :] = mx
        market_shares[i, 0:mx.shape[0]] = blp95.data.loc[blp95.data.year == mkt_id, 'share'].values
        log_market_shares[i, 0:mx.shape[0]] = np.log(market_shares[i, 0:mx.shape[0]])

    prod_char = torch.as_tensor(prod_char, device=device, dtype=t_type)
    mdraws = torch.as_tensor(mdraws, device=device, dtype=t_type)
    mweights = torch.as_tensor(mweights, device=device, dtype=t_type)
    market_shares = torch.as_tensor(market_shares, device=device, dtype=t_type)
    log_market_shares = torch.as_tensor(log_market_shares, device=device, dtype=t_type)

    # we need model_id's to aggregate the moments later
    model_ids = blp95.data[blp95.model_id_col].values
    uniq_model_ids = np.unique(model_ids)
    model_ids_to_rows = np.repeat(uniq_model_ids[:, None], model_ids.shape[0], axis=1)
    model_ids_to_rows = (model_ids_to_rows == model_ids).astype(np.float32)
    model_ids_to_rows = np.tile(model_ids_to_rows, 2)
    model_ids_to_rows = torch.as_tensor(model_ids_to_rows, device=device, dtype=t_type)

    # joint ownership matrix
    own_mat = torch.as_tensor(blp95.own_mat, device=device, dtype=t_type)

    # non-linear parameters to optimize
    theta = torch.as_tensor(blp95.random_component_values, device=device, dtype=t_type).requires_grad_(True)

    delta_0 = None

    optimizer = torch.optim.LBFGS(
        [theta],
        max_iter=200,
        line_search_fn='strong_wolfe')

    def obj(full_output=False, verbose=True, grad=True):
        optimizer.zero_grad()
        res = gmm_loss(
            log_market_shares,
            prod_char,
            market_masks,
            own_mat,
            model_ids_to_rows,
            beta_mat,
            xw_mat,
            z_mat,
            mdraws,
            mweights,
            theta,
            weights_0,
            delta_0=delta_0,
            full_output=full_output,
            atol=atol,
            rtol=rtol,
        )
        res[0].backward()
        if verbose:
            print(f'GMM loss: {res[0].item():.6f}')
        if full_output:
            return res
        else:
            return res[0]

    # get some initial GMM weights and some initial deltas for the inversion
    __, delta_0, __, weights_0 = obj(full_output=True)
    delta_0 = delta_0.detach()

    # update projection matrices
    proj_z = z_mat @ weights_0 @ z_mat.T
    xw_t_xw_inv = torch.inverse(xw_mat.T @ proj_z @ xw_mat)
    beta_mat = xw_t_xw_inv @ xw_mat.T @ proj_z

    # GMM step 1
    print('=' * 80)
    t_step = time.time()
    optimizer.step(obj)
    t_step = time.time() - t_step
    print(f'time: {t_step:.2f} secs')

    # update GMM weights
    __, __, __, weights_0 = obj(full_output=True)

    # update projection matrices
    proj_z = z_mat @ weights_0 @ z_mat.T
    xw_t_xw_inv = torch.inverse(xw_mat.T @ proj_z @ xw_mat)
    beta_mat = xw_t_xw_inv @ xw_mat.T @ proj_z

    # GMM step 2
    print('=' * 80)
    t_step = time.time()
    optimizer.step(obj)
    t_step = time.time() - t_step
    print(f'time: {t_step:.2f} secs')

    # get linear parameters
    __, __, beta, __ = obj(full_output=True)

    print('=' * 80)
    for t0 in theta:
        print(f'{t0:8.4f}')
    for b0 in beta:
        print(f'{b0:8.4f}')
    print('=' * 80)

    t_tot = time.time() - t_tot
    print(f'total time: {t_tot:.2f} secs')
