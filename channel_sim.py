import yaml
import numpy as np
import cupy as cp
from einops import rearrange
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from .simulator.ray import RayGroup
from .simulator.ue import UEGroup
from .simulator.background import Background


class Channel:
    def __init__(self, scenario_name, simulation_name, OFDM_params, array_params):
        self.background = Background.load(scenario_name=scenario_name, simulation_name=simulation_name)
        self.bs_name_list = self.background.get_bs_name_list()
        self.bs_attitude_dict = self.background.get_bs_attitude_dict()
        self.ue = UEGroup.load(scenario_name=scenario_name, simulation_name=simulation_name)
        self.ue_name_list = self.ue.get_ue_name_list()
        self.ue_time_idx_dict = self.ue.get_ue_time_idx_dict()
        self.ray = RayGroup.load(scenario_name=scenario_name, simulation_name=simulation_name)
        self.wavelength = self.ray.wavelength
        # OFDM parameters
        self.fft_size = OFDM_params['fft_size']
        self.subcarrier_spacing = OFDM_params['subcarrier_spacing']
        self.sampling_rate = self.fft_size * self.subcarrier_spacing
        start = -self.fft_size / 2 if self.fft_size % 2 == 0 else -(self.fft_size - 1) / 2
        limit = self.fft_size / 2 if self.fft_size % 2 == 0 else (self.fft_size - 1) / 2 + 1
        self._frequencies = np.arange(start=start, stop=limit) * self.subcarrier_spacing
        # Array parameters
        self.array_params = array_params
        self.array_position = {}
        for n in array_params:
            position = None
            if array_params[n]['type'] == 'array':
                position = np.array(array_params[n]['position']).astype(np.float64)
            elif array_params[n]['type'] == 'planar':
                num_rows = array_params[n]['num_rows']
                num_cols = array_params[n]['num_cols']
                vertical_spacing = array_params[n]['vertical_spacing']
                horizontal_spacing = array_params[n]['horizontal_spacing']
                row_coord = (np.arange(start=0, stop=num_rows) - (num_rows - 1) / 2) * vertical_spacing
                col_coord = (np.arange(start=0, stop=num_cols) - (num_cols - 1) / 2) * horizontal_spacing
                yz = np.stack(arrays=(np.repeat(col_coord[:, np.newaxis], repeats=num_rows, axis=1),
                                      np.repeat(row_coord[np.newaxis, :], repeats=num_cols, axis=0)), axis=2)
                yz = np.reshape(yz, newshape=(-1, yz.shape[2]))
                position = np.concatenate((np.zeros((yz.shape[0], 1)), yz), axis=1).astype(np.float64)
            self.array_position[n] = position * self.wavelength
        bs_attitude_list = list(self.bs_attitude_dict.items())
        bs_attitude = np.stack([x[1] for x in bs_attitude_list], axis=0)
        self._bs_rot_mat = self._rotation_matrix(bs_attitude)
        bs_array = np.einsum('brc, ac -> bar', self._bs_rot_mat, self.array_position['bs'])  # bs, ant, coord
        self._bs_array_dict = {}
        for idx, (name, _) in enumerate(bs_attitude_list):
            self._bs_array_dict[name] = bs_array[idx, :, :]
        # Simulation step
        self._cur_time_idx = 0
        self._data_loading_step = 1
        self._loaded_ch_data = None
        self._loaded_time_idx_range = None
        self._sim_bs_name_list = None
        self._sim_ue_name_list = None
        self._sim_sparse_channel = False
        self._sim_cov_mat = False
        self._sim_sparse_cov_mat = False
        # Cupy memory management
        self._cp_mempool = cp.get_default_memory_pool()

    def init_time_step_simulation(self, start_time_idx, data_loading_step=1, bs_name_list=None, ue_name_list=None,
                                  sparse_channel=False, cov_mat=False, sparse_cov_mat=False):
        self._cur_time_idx = start_time_idx
        self._data_loading_step = data_loading_step
        self._sim_bs_name_list = bs_name_list
        self._sim_ue_name_list = ue_name_list
        self._sim_sparse_channel = sparse_channel
        self._sim_cov_mat = cov_mat
        self._sim_sparse_cov_mat = sparse_cov_mat
        self._loaded_ch_data = None
        self._loaded_time_idx_range = None

    def __iter__(self):
        return self

    def __next__(self):
        if (self._loaded_time_idx_range is None or
                not (self._loaded_time_idx_range[0] <= self._cur_time_idx < self._loaded_time_idx_range[1])):
            if self._cur_time_idx >= len(self.ray.time):
                raise StopIteration
            end_time_idx = min(self._cur_time_idx + self._data_loading_step, len(self.ray.time))
            self._loaded_time_idx_range = (self._cur_time_idx, end_time_idx)
            self._loaded_ch_data = self.ray.get_data_tensor(self._sim_bs_name_list, self._sim_ue_name_list, self._loaded_time_idx_range)
        if self._loaded_ch_data is None:
            return {'time_idx': self._cur_time_idx, 'ue_name_list': None}
        time_idx = self._cur_time_idx - self._loaded_time_idx_range[0]
        ue_mask = self._loaded_ch_data['ue_mask'][:, time_idx]
        bs_name_list = self._loaded_ch_data['bs_name_list']
        ue_name_list = list(np.array(self._loaded_ch_data['ue_name_list'])[ue_mask])
        ue_att = self._loaded_ch_data['ue_att'][ue_mask, :][:, time_idx:time_idx + 1]
        a = self._loaded_ch_data['a'][:, ue_mask, :][:, :, time_idx:time_idx + 1]
        phi_r = self._loaded_ch_data['phi_r'][:, ue_mask, :][:, :, time_idx:time_idx + 1]
        phi_t = self._loaded_ch_data['phi_t'][:, ue_mask, :][:, :, time_idx:time_idx + 1]
        theta_r = self._loaded_ch_data['theta_r'][:, ue_mask, :][:, :, time_idx:time_idx + 1]
        theta_t = self._loaded_ch_data['theta_t'][:, ue_mask, :][:, :, time_idx:time_idx + 1]
        tau = self._loaded_ch_data['tau'][:, ue_mask, :][:, :, time_idx:time_idx + 1]
        ch_data = {'bs_name_list': bs_name_list, 'ue_att': ue_att, 'a': a, 'phi_r': phi_r, 'phi_t': phi_t,
                   'theta_r': theta_r, 'theta_t': theta_t, 'tau': tau}
        proc_ch = self.process_ch_data(ch_data, self._sim_sparse_channel, self._sim_cov_mat, self._sim_sparse_cov_mat)
        ue_pos = self._loaded_ch_data['ue_pos'][ue_mask, :][:, time_idx]
        ret = {'time_idx': self._cur_time_idx, 'ue_name_list': ue_name_list, 'bs_name_list': bs_name_list, 'ue_pos': ue_pos}
        ret['ofdm_ch'] = proc_ch['ofdm_ch'][:, :, 0]  # bs, ue, bs_ant, ue_ant, freq
        ret['direction_vector'] = proc_ch['direction_vector'][:, :, 0] # bs, ue, path, 3
        ret['a'] = a[:, :, 0]  # bs, ue, path
        ret['tau'] = tau[:, :, 0]  # bs, ue, path
        if self._sim_sparse_channel:  # bs, ue, bs_ant_col, bs_ant_row, ue_ant, freq
            ret['sparse_ch'] = proc_ch['sparse_ch'][:, :, 0]
        if self._sim_cov_mat:  # bs, ue, bs_ant, bs_ant
            ret['cov_mat'] = proc_ch['cov_mat'][:, :, 0]
        if self._sim_sparse_cov_mat:  # bs, ue, bs_ant, bs_ant
            ret['sparse_cov_mat'] = proc_ch['sparse_cov_mat'][:, :, 0]
        self._cur_time_idx += 1
        return ret

    def get_sch_pos(self, ue, time_idx):
        time_idx_range = (time_idx, time_idx + 1)
        data = self.ray.get_data_tensor(bs_name_list=None, ue_name_list=[ue], time_idx_range=time_idx_range)
        proc_ch = self.process_ch_data(data, sparse_channel=True)
        sch = proc_ch['sparse_ch'][:, 0, 0, :, 0, 0, :]  # bs, bs_ant, freq
        pos = data['ue_pos'][0, 0, :2]  # xy
        return sch, pos

    def get_ch_pos(self, ue, time_idx):
        time_idx_range = (time_idx, time_idx + 1)
        data = self.ray.get_data_tensor(bs_name_list=None, ue_name_list=[ue], time_idx_range=time_idx_range)
        proc_ch = self.process_ch_data(data, sparse_channel=True)
        ch = proc_ch['ofdm_ch'][:, 0, 0, :, 0, :]  # bs, bs_ant, freq
        pos = data['ue_pos'][0, 0, :2]  # xy
        return ch, pos

    def get_channel(self, bs_name_list, ue_name_list, time_idx_range, sparse_channel=False,
                    cov_mat=False, sparse_cov_mat=False):
        data = self.ray.get_data_tensor(bs_name_list, ue_name_list, time_idx_range)
        if data is None:
            return None
        proc_ch = self.process_ch_data(data, sparse_channel, cov_mat, sparse_cov_mat)
        ret = {'bs_name_list': data['bs_name_list'], 'ue_name_list': data['ue_name_list'],
               'ue_mask': data['ue_mask'], 'mask': data['mask'], 'tau': data['tau'], 'ue_pos': data['ue_pos']}
        ret['ofdm_ch'] = proc_ch['ofdm_ch']  # bs, ue, time, bs_ant, ue_ant, freq
        if sparse_channel:  # bs, ue, time, bs_ant_col, bs_ant_row, ue_ant, freq
            ret['sparse_ch'] = proc_ch['sparse_ch']
        if cov_mat:  # bs, ue, time, bs_ant, bs_ant
            ret['cov_mat'] = proc_ch['cov_mat']
        if sparse_cov_mat:  # bs, ue, time, bs_ant, bs_ant
            ret['sparse_cov_mat'] = proc_ch['sparse_cov_mat']
        return ret

    def process_ch_data(self, data, sparse_channel=False, cov_mat=False, sparse_cov_mat=False):
        ch, dv = self.cal_ofdm_ch(data)  # ch (bs, ue, time, bs_ant, ue_ant, freq), dv (bs, ue, time, path, 3)
        ret = {'ofdm_ch': ch.get(), 'direction_vector': dv.get()}
        if sparse_channel:
            s_ch = self.cal_sparse_domain_channel(ch)
            ret['sparse_ch'] = s_ch.get()  # bs, ue, time, bs_ant_col, bs_ant_row, ue_ant, freq
        if cov_mat or sparse_cov_mat:
            cm = self.cal_cov_mat(ch)
            if cov_mat:
                ret['cov_mat'] = cm.get()  # bs, ue, time, bs_ant, bs_ant
            if sparse_cov_mat:
                scm = self.cal_sparse_cov_mat(cm)
                ret['sparse_cov_mat'] = scm.get()  # bs, ue, time, bs_ant, bs_ant
        # self._cp_mempool.free_all_blocks()
        return ret

    def cal_ofdm_ch(self, data):
        bs_names = data['bs_name_list']
        ue_att = data['ue_att']
        theta_r, theta_t, phi_r, phi_t = cp.array(data['theta_r']), cp.array(data['theta_t']), cp.array(data['phi_r']), cp.array(data['phi_t'])
        bs_array = []
        for name in bs_names:
            bs_array.append(self._bs_array_dict[name])
        bs_array = cp.array(np.stack(bs_array, axis=0))  # bs, bs_ant, coord
        ue_rot_mat = cp.array(self._rotation_matrix(ue_att))  # ue, time, row, col
        ue_array = cp.einsum('utrc, ac -> utar', ue_rot_mat, cp.array(self.array_position['ue']))  # ue, time, ue_ant, coord
        k_bs = cp.stack(tup=[cp.sin(theta_t) * cp.cos(phi_t),
                             cp.sin(theta_t) * cp.sin(phi_t),
                             cp.cos(theta_t)], axis=-1)  # bs, ue, time, path, coord
        k_ue = cp.stack(tup=[cp.sin(theta_r) * cp.cos(phi_r),
                             cp.sin(theta_r) * cp.sin(phi_r),
                             cp.cos(theta_r)], axis=-1)  # bs, ue, time, path, coord
        bs_phase_shifts = cp.einsum('bic, butpc -> butpi', bs_array, k_bs)  # bs, ue, time, path, bs_ant(i)
        ue_phase_shifts = cp.einsum('utjc, butpc -> butpj', ue_array, k_ue)  # bs, ue, time, path, ue_ant(j)
        phase_shifts = bs_phase_shifts[:, :, :, :, :, cp.newaxis] + ue_phase_shifts[:, :, :, :, cp.newaxis, :]  # bs, ue, time, path, bs_ant, ue_ant
        phase_shifts = 2 * cp.pi * phase_shifts / self.wavelength
        array_coef = cp.exp(1j * phase_shifts)  # bs, ue, time, path, bs_ant, ue_ant
        # Compute direction vector
        dir_vec = np.einsum('brc, butpc -> butpr', cp.transpose(self._bs_rot_mat, axes=(0, 2, 1)), k_bs)  # bs, ue, time, path, coord
        # Compute OFDM channel
        a = cp.array(data['a'])  # bs, ue, time, path
        tau = cp.array(data['tau'])  # bs, ue, time, path
        frequencies = cp.array(self._frequencies)  # freq
        n_bs, n_ue, n_time, n_path = tau.shape
        n_freq = frequencies.shape[0]
        n_bs_ant, n_ue_ant = array_coef.shape[4], array_coef.shape[5]
        ch = cp.zeros((n_bs, n_ue, n_time, n_bs_ant, n_ue_ant, n_freq)).astype(cp.complex64)  # bs, ue, time, bs_ant, ue_ant, freq
        for p in range(n_path):
            freq_coef = cp.exp(- 2j * cp.pi * frequencies * tau[:, :, :, p, cp.newaxis])  # bs, ue, time, frequency
            tmp = (a[:, :, :, p, cp.newaxis, cp.newaxis, cp.newaxis] * freq_coef[:, :, :, cp.newaxis, cp.newaxis, :]
                   * array_coef[:, :, :, p, :, :, cp.newaxis])
            ch += tmp
        return ch, dir_vec

    def cal_sparse_domain_channel(self, ch):
        if self.array_params['bs']['type'] == 'planar':
            num_rows = self.array_params['bs']['num_rows']
            num_cols = self.array_params['bs']['num_cols']
            ch = rearrange(ch, 'b u t (bac bar) ua f -> b u t bac bar ua f', bac=num_cols, bar=num_rows)  # bs, ue, time, bs_ant_col, bs_ant_row, ue_ant, freq
            ch = cp.fft.fftshift(cp.fft.fft(ch, axis=-4, norm='forward'), axes=-4)  # column fft
            ch = cp.fft.fftshift(cp.fft.fft(ch, axis=-3, norm='forward'), axes=-3)  # row fft
            ch = cp.fft.ifft(cp.fft.ifftshift(ch, axes=-1), axis=-1, norm='backward')  # freq ifft
            # ch = cp.fft.fft(ch, axis=-4, norm='forward')  # column fft
            # ch = cp.fft.fft(ch, axis=-3, norm='forward')  # row fft
            # ch = cp.fft.ifft(ch, axis=-1, norm='backward')  # freq ifft
        else:
            raise Exception('BS should be a planar array.')
        return ch

    def cal_cov_mat(self, ch):
        ch = ch[:, :, :, :, 0, :]  # bs, ue, time, bs_ant, freq
        cov_mat = cp.matmul(cp.conjugate(ch), cp.transpose(ch, axes=(0, 1, 2, 4, 3))) / self.fft_size  # bs, ue, time, bs_ant, bs_ant
        return cov_mat

    def cal_sparse_cov_mat(self, cov_mat):
        if self.array_params['bs']['type'] == 'planar':
            num_rows = self.array_params['bs']['num_rows']
            num_cols = self.array_params['bs']['num_cols']
            cov_mat = rearrange(cov_mat, 'b u t (bac1 bar1) (bac2 bar2) -> b u t bac1 bar1 bac2 bar2',
                                bac1=num_cols, bar1=num_rows, bac2=num_cols, bar2=num_rows)
            cov_mat = cp.fft.fft(cov_mat, axis=3, norm='forward')
            cov_mat = cp.fft.fft(cov_mat, axis=4, norm='forward')
            cov_mat = cp.fft.fft(cov_mat, axis=5, norm='forward')
            cov_mat = cp.fft.fft(cov_mat, axis=6, norm='forward')
        else:
            raise Exception('BS should be a planar array.')
        cov_mat = rearrange(cov_mat, 'b u t bac1 bar1 bac2 bar2 -> b u t (bac1 bar1) (bac2 bar2)',
                            bac1=num_cols, bar1=num_rows, bac2=num_cols, bar2=num_rows)
        return cov_mat

    def plot_sparce_domain_channel(self, ch, start_time_idx, time_idx_step, n_time_plot_row, n_time_plot_col, scale=3):
        ch = ch[0, 0, :, :, 0, 0, :]  # time, azimuth, delay
        ch = cp.fft.fftshift(ch, axes=-2)
        ch = 20 * cp.log10(cp.abs(ch)).get()
        num_plot = n_time_plot_row * n_time_plot_col
        time_index_list = list(np.arange(start_time_idx, start_time_idx + time_idx_step * num_plot, time_idx_step))
        fig = plt.figure(figsize=(scale * n_time_plot_col, scale * n_time_plot_row))
        gs = GridSpec(nrows=n_time_plot_row, ncols=n_time_plot_col)
        for idx, t_idx in enumerate(time_index_list):
            ax = fig.add_subplot(gs[idx])
            ax.set_title(f'{t_idx}')
            c = ch[t_idx, :, :]
            ax.pcolor(c, shading='auto')
        plt.show()

    def _rotation_matrix(self, angles):
        # angles : (z, y, x) intrinsic rotation angle
        angles = cp.array(angles)
        a = angles[..., 0]
        b = angles[..., 1]
        c = angles[..., 2]

        cos_a = cp.cos(a)
        cos_b = cp.cos(b)
        cos_c = cp.cos(c)
        sin_a = cp.sin(a)
        sin_b = cp.sin(b)
        sin_c = cp.sin(c)

        r_11 = cos_a * cos_b
        r_12 = cos_a * sin_b * sin_c - sin_a * cos_c
        r_13 = cos_a * sin_b * cos_c + sin_a * sin_c
        r_1 = cp.stack((r_11, r_12, r_13), axis=-1)

        r_21 = sin_a * cos_b
        r_22 = sin_a * sin_b * sin_c + cos_a * cos_c
        r_23 = sin_a * sin_b * cos_c - cos_a * sin_c
        r_2 = cp.stack((r_21, r_22, r_23), axis=-1)

        r_31 = -sin_b
        r_32 = cos_b * sin_c
        r_33 = cos_b * cos_c
        r_3 = cp.stack((r_31, r_32, r_33), axis=-1)

        rot_mat = cp.stack((r_1, r_2, r_3), axis=-2)
        rot_mat = rot_mat.get()
        return rot_mat
