import pandas as pd
import numpy as np


class BLP95Model(object):
    def __init__(self, data_file, unobs_file):
        self.data = pd.read_csv(data_file)
        self.mkt_id_col = 'year'
        self.firm_id_col = 'firm_id'
        self.model_id_col = 'model_id'

        market_cnts = self.data.groupby(self.mkt_id_col, as_index=False).model_id.count()
        self.market_names = market_cnts[self.mkt_id_col].values
        self.market_sizes = market_cnts[self.model_id_col].values

        self.random_component_names = [
            'price', 'const', 'hpwt', 'air', 'mpd', 'space']

        self.random_component_values = [
            43.501, 3.612, 4.628, 1.818, 1.050, 2.056]

        self.max_market_size = np.max(self.market_sizes)
        self.num_markets = len(self.data.year.unique())
        self.num_prod_char = len(self.random_component_names)

        self.market_masks = np.zeros((self.num_markets, self.max_market_size))
        for i in range(len(self.market_masks)):
            self.market_masks[i, 0:self.market_sizes[i]] = 1

        self.__load_unobs_draws(unobs_file)

        self.delta_regressor_cols = [
            'const', 'hpwt', 'air', 'mpd', 'space']

        self.cost_regressor_cols = [
            'const', 'loghpwt', 'air', 'logmpg', 'logspace', 'trend']

        self.xi_moment_cols = [
            'const',
            'demeaned_hpwt',
            'demeaned_air',
            'demeaned_mpd',
            'demeaned_space',
            'demeaned_demand_firm_const',
            'demeaned_demand_firm_hpwt',
            'demeaned_demand_firm_air',
            'demeaned_demand_firm_mpd',
            'demeaned_demand_rival_const',
            'demeaned_demand_rival_hpwt',
            'demeaned_demand_rival_air',
            'demeaned_demand_rival_mpd',
        ]

        self.omega_moment_cols = [
            'const',
            'demeaned_loghpwt',
            'demeaned_air',
            'demeaned_logmpg',
            'demeaned_logspace',
            'demeaned_trend',
            'demeaned_supply_firm_const',
            'demeaned_supply_firm_loghpwt',
            'demeaned_supply_firm_air',
            'demeaned_supply_firm_logmpg',
            'demeaned_supply_firm_logspace',
            'demeaned_supply_firm_trend',
            'demeaned_supply_rival_const',
            'demeaned_supply_rival_loghpwt',
            'demeaned_supply_rival_air',
            'demeaned_supply_rival_logmpg',
            'demeaned_supply_rival_logspace',
            'demeaned_mpd',
        ]

        self.own_mat = self.__create_ownership_matrices()

    def __create_ownership_matrices(self):
        own_mat = np.zeros((self.num_markets, self.max_market_size, self.max_market_size))
        for m, mkt_id in enumerate(self.market_names):
            mkt_data = self.data.loc[self.data[self.mkt_id_col] == mkt_id]
            firm_ids = mkt_data[self.firm_id_col].values
            for i in range(len(firm_ids)):
                for j in range(i, len(firm_ids)):
                    if firm_ids[i] == firm_ids[j]:
                        own_mat[m, i, j] = own_mat[m, j, i] = 1
        return own_mat

    def __load_unobs_draws(self, filename):
        unobs_colnames = [
            'income', 'const', 'hpwt', 'air', 'mpd', 'space']
        unobs_weight_col = 'weight'
        data = pd.read_csv(filename)
        draws = data[unobs_colnames].values
        self.weights = data[unobs_weight_col].values
        self.weights /= draws.shape[0]

        log_income_means = self.data.groupby(self.mkt_id_col)['log_income_mean'].max().values[:, None]
        self.draws = np.tile(draws, (self.num_markets, 1, 1))
        self.draws[:, :, 0] *= 1.72
        self.draws[:, :, 0] += log_income_means
        self.draws[:, :, 0] = -1 / np.exp(self.draws[:, :, 0])
        self.draws = self.draws.swapaxes(1, 2)
