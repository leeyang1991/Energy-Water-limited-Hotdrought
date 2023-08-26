# coding=utf-8
from __init__ import *

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 8
centimeter_factor = 1 / 2.54

this_root = '/Users/liyang/Projects_data/Energy_water_hotdrought/'
data_root = this_root + 'data/'
results_root = this_root + 'results/'
temp_root = this_root + 'temp/'

global_drought_type_list = ['normal-drought', 'hot-drought']
global_drought_type_color_dict = {
    'normal-drought': 'blue',
    'hot-drought': 'red',
}
global_ELI_class = ['Energy-Limited', 'Water-Limited']
global_ELI_class_color_dict = {
    'Energy-Limited': 'blue',
    'Water-Limited': 'red',
}
global_AI_class = ['Humid', 'Arid']
global_land_tif = join(this_root,'conf/land.tif')
global_land_tif_reproj = join(this_root,'conf/land_reproj.tif')
global_year_range = '1982-2020'
global_start_year,global_end_year = global_year_range.split('-')
global_start_year = int(global_start_year)
global_end_year = int(global_end_year)
global_year_range_list = list(range(global_start_year,global_end_year+1))
# data_path_dict = Meta_information().path(global_year_range)
global_gs = list(range(5,11))

global_lc_list = ('deciduous', 'evergreen', 'grass', 'shrubs')
global_lc_marker_dict = {
    'deciduous': 'o',
    'evergreen': 's',
    'grass': 'v',
    'shrubs': 'D',
}
global_koppen_list = ('arid', 'cold arid', 'cold humid', 'hot arid', 'hot humid')
global_koppen_color_dict = {
    'arid': '#EB6100',
    'cold arid': '#601986',
    'cold humid': 'b',
    'hot arid': 'r',
    'hot humid': 'g',
}
global_ELI_class_list = ('Energy-Limited', 'Water-Limited')
global_AI_class_list = ('Humid', 'Arid')
global_threshold = 0.05

global_VIs_list = ['NDVI','VOD-anomaly','CSIF-anomaly']
global_VIs_origin_list = ['NDVI-origin','VOD-origin','CSIF-origin']
global_VIs_year_range_dict = {
    'NDVI': '1982-2015',
    'NDVI-origin': '1982-2015',
    'VOD-origin': '2003-2015',
    'VOD-anomaly': '2003-2015',
    'CSIF-origin': '2001-2015',
    'CSIF-anomaly': '2001-2015',
    'VOD-k-band-origin': '1988-2015',
    'VOD-k-band-anomaly': '1988-2015',
}
global_color_list = [
    '#844000',
    '#fc9831',
    '#fffbd4',
    '#86b9d2',
    '#064c6c',
]
global_cmap = T.cmap_blend(global_color_list,)
global_cmap_r = T.cmap_blend(global_color_list[::-1])
global_spei_list = ['spei03', 'spei06', 'spei09', 'spei12',
                     'spei15','spei18', 'spei21', 'spei24',
                     ]
global_all_spei_list = [
    'spei01', 'spei02', 'spei03', 'spei04', 'spei05', 'spei06', 'spei07', 'spei08', 'spei09', 'spei10', 'spei11', 'spei12',
    'spei13', 'spei14', 'spei15', 'spei16', 'spei17', 'spei18', 'spei19', 'spei20', 'spei21', 'spei22', 'spei23', 'spei24',
                     ]
global_all_spi_list = [i.replace('spei','spi') for i in global_all_spei_list]
global_ELI_bins = np.arange(-0.8, 0.75, .05)
global_lag_list = list(range(0, 7))
global_drought_season_list = ['early','mid','late']
global_drought_season_dict = {
    'early': [5,6],
    'mid': [7,8],
    'late': [9,10],
}

global_season_mon_dict = {
    5: 'early',
    6: 'early',
    7: 'mid',
    8: 'mid',
    9: 'late',
    10: 'late',
    1:'other',
    2:'other',
    3:'other',
    4:'other',
    11:'other',
    12:'other',
}

class Meta_information:

    def __init__(self):
        pass

    def path(self,year_range=global_year_range):
        vars_info_dic = {
            'SPEI': {
                'path':join(data_root, 'SPEI/per_pix_clean',year_range),
                'path_type':'multi-files',
            },
            'CCI-SM-origin': {
                'path': join(data_root, f'CCI_SM_v7/per_pix/{year_range}'),
                'path_type': 'dir',
            },
            'CCI-SM-anomaly': {
                'path': join(data_root, f'CCI_SM_v7/anomaly/{year_range}'),
                'path_type': 'dir',
            },
            'CCI-SM-anomaly_detrend': {
                'path': join(data_root, f'CCI_SM_v7/anomaly_detrend/{year_range}'),
                'path_type': 'dir',
            },
            'ERA-SM-anomaly': {
                'path': join(data_root, f'ERA-SM/anomaly/{year_range}'),
                'path_type': 'dir',
            },
            'GLEAM-SMRoot-origin': {
                'path': join(data_root, f'GLEAM_SMRoot/per_pix/{year_range}'),
                'path_type': 'dir',
            },
            'GLEAM-SMRoot-anomaly': {
                'path': join(data_root, f'GLEAM/anomaly/{year_range}'),
                'path_type': 'dir',
            },
            'GLEAM-SMRoot-anomaly_detrend': {
                'path': join(data_root, f'GLEAM/anomaly_detrend/{year_range}'),
                'path_type': 'dir',
            },
            'SPI': {
                'path': join(data_root, 'SPI/per_pix',year_range),
                'path_type': 'dir',
            },
            'NDVI-origin': {
                'path': join(data_root, 'NDVI4g/per_pix', year_range),
                'path_type': 'dir',
            },
            'NDVI-anomaly_detrend': {
                'path': join(data_root, 'NDVI4g/per_pix_anomaly_detrend', year_range),
                'path_type': 'dir',
            },
            'CSIF-origin': {
                'path': join(data_root, 'CSIF/per_pix', year_range),
                'path_type': 'dir',
            },
            'CSIF-anomaly': {
                'path': join(data_root, 'CSIF/per_pix_anomaly', year_range),
                'path_type': 'dir',
            },
            'VOD-origin': {
                'path': join(data_root, 'VOD_Xband/per_pix', year_range),
                'path_type': 'dir',
            },
            'VOD-anomaly': {
                'path': join(data_root, 'VOD_Xband/per_pix_anomaly_detrend', year_range),
                'path_type': 'dir',
            },
            'VOD-k-band-origin': {
                'path': join(data_root, 'VOD_Kband/per_pix', year_range),
                'path_type': 'dir',
            },
            'VOD-k-band-anomaly': {
                'path': join(data_root, 'VOD_Kband/detrend', year_range),
                'path_type': 'dir',
            },
            'Temperature-origin': {
                'path': join(data_root, f'CRU_tmp/per_pix/{year_range}'),
                'path_type': 'dir',
            },
            'Temperature-origin_detrend': {
                'path': join(data_root, f'CRU_tmp/per_pix_detrend/{year_range}'),
                'path_type': 'dir',
            },
            'Temperature-anomaly': {
                'path': join(data_root, f'CRU_tmp/anomaly/{year_range}'),
                'path_type': 'dir',
            },
            'Temperature-anomaly_detrend': {
                'path': join(data_root, f'CRU_tmp/anomaly_detrend/{year_range}'),
                'path_type': 'dir',
            },
            'Precipitation-detrend': {
                'path': join(data_root, f'CRU_precip/detrend/{year_range}/precip.npy'),
                'path_type': 'file',
            },
            'Precipitation-origin': {
                'path': join(data_root, f'CRU_precip/per_pix/{year_range}'),
                'path_type': 'dir',
            },
            'Precipitation-anomaly': {
                'path': join(data_root, f'CRU_precip/anomaly/{year_range}'),
                'path_type': 'dir',
            },
            'Precipitation-anomaly_detrend': {
                'path': join(data_root, f'CRU_precip/anomaly_detrend/{year_range}'),
                'path_type': 'dir',
            },
            'Radiation': {
                'path': join(data_root, f'Terraclimate/srad/detrend/{year_range}/srad.npy'),
                'path_type': 'file',
            },
            'Terra-ET': {
                'path': join(data_root, f'Terraclimate/aet/detrend/{year_range}/aet.npy'),
                'path_type': 'file',
            },
            'GLEAM-ET': {
                'path': join(data_root, f'GLEAM_ET/detrend/{year_range}/GLEAM_ET.npy'),
                'path_type': 'file',
            },
            'VPD': {
                'path': join(data_root, f'VPD/anomaly/{year_range}/VPD.npy'),
                'path_type': 'file',
            },
            'VPD-origin': {
                'path': join(data_root, f'VPD/per_pix/{year_range}/VPD.npy'),
                'path_type': 'file',
            },
            'GOME2_SIF-origin': {
                'path': join(data_root, f'GOME2_SIF/per_pix/{year_range}'),
                'path_type': 'dir',
            },
            'GOME2_SIF-anomaly': {
                'path': join(data_root, f'GOME2_SIF/detrend/{year_range}'),
                'path_type': 'dir',
            },
            'ERA_Ta2m-origin': {
                'path': join(data_root, 'ERA_2m_T/perpix', year_range),
                'path_type': 'dir',
            },
            'ERA_Ta2m-anomaly': {
                'path': join(data_root, 'ERA_2m_T/anomaly', year_range),
                'path_type': 'dir',
            },
            'ERA_precip-origin': {
                'path': join(data_root, 'ERA_Precip/perpix', year_range),
                'path_type': 'dir',
            },
            'ERA_precip-anomaly': {
                'path': join(data_root, 'ERA_Precip/anomaly', year_range),
                'path_type': 'dir',
            },
            'GPCC-origin': {
                'path': join(data_root, 'GPCC/perpix', year_range),
                'path_type': 'dir',
            },
            'GPCC-anomaly': {
                'path': join(data_root, 'GPCC/anomaly', year_range),
                'path_type': 'dir',
            },
            'BEST-anomaly': {
                'path': join(data_root, 'Berkeley Earth Surface Temperatures/perpix', year_range),
                'path_type': 'dir',
            },
        }
        return vars_info_dic

    def load_data(self, var_i, year_range=global_year_range):
        data_path_dict = Meta_information().path(year_range)
        data_path = data_path_dict[var_i]['path']
        path_type = data_path_dict[var_i]['path_type']
        if path_type == 'file':
            spatial_dict = T.load_npy(data_path)
        elif path_type == 'dir':
            spatial_dict = T.load_npy_dir(data_path)
        elif path_type == 'multi-files':
            spatial_dict = {}
            for f in T.listdir(data_path):
                print(f'loading {f}')
                key = f.split('.')[0]
                spatial_dict_i = T.load_npy(join(data_path, f))
                spatial_dict[key] = spatial_dict_i
        else:
            raise ValueError('path_type not recognized')
        return spatial_dict


class Load_Data:

    def __init__(self):

        pass

    def SPEI(self,year_range=global_year_range):
        data_path = join(data_root, 'SPEI/per_pix_clean', year_range)
        path_type = 'multi-files'
        var_name = 'SPEI'
        return self.__load_data(data_path, path_type), var_name
    
    def CCI_SM_origin(self,year_range=global_year_range):
        data_path = join(data_root, f'CCI_SM_v7/per_pix/{year_range}')
        path_type = 'dir'
        var_name = 'CCI-SM-origin'
        return self.__load_data(data_path, path_type), var_name

    def CCI_SM_anomaly(self,year_range=global_year_range):
        data_path = join(data_root, f'CCI_SM_v7/anomaly/{year_range}')
        path_type = 'dir'
        var_name = 'CCI-SM-anomaly'
        return self.__load_data(data_path, path_type), var_name

    def CCI_SM_anomaly_detrend(self,year_range=global_year_range):
        data_path = join(data_root, f'CCI_SM_v7/anomaly_detrend/{year_range}')
        path_type = 'dir'
        var_name = 'CCI-SM-anomaly_detrend'
        return self.__load_data(data_path, path_type), var_name

    def ERA_SM_anomaly(self,year_range=global_year_range):
        data_path = join(data_root, f'ERA-SM/anomaly/{year_range}')
        path_type = 'dir'
        var_name = 'ERA-SM-anomaly'
        return self.__load_data(data_path, path_type), var_name

    def GLEAM_SMRoot_origin(self,year_range=global_year_range):
        data_path = join(data_root, f'GLEAM_SMRoot/per_pix/{year_range}')
        path_type = 'dir'
        var_name = 'GLEAM-SMRoot-origin'
        return self.__load_data(data_path, path_type), var_name

    def GLEAM_SMRoot_anomaly(self,year_range=global_year_range):
        data_path = join(data_root, f'GLEAM/anomaly/{year_range}')
        path_type = 'dir'
        var_name = 'GLEAM-SMRoot-anomaly'
        return self.__load_data(data_path, path_type), var_name

    def GLEAM_SMRoot_anomaly_detrend(self,year_range=global_year_range):
        data_path = join(data_root, f'GLEAM/anomaly_detrend/{year_range}')
        path_type = 'dir'
        var_name = 'GLEAM-SMRoot-anomaly_detrend'
        return self.__load_data(data_path, path_type), var_name

    def SPI(self,year_range=global_year_range):
        data_path = join(data_root, 'SPI/per_pix',year_range)
        path_type = 'dir'
        var_name = 'SPI'
        return self.__load_data(data_path, path_type), var_name

    def NDVI_origin(self,year_range=global_year_range):
        data_path = join(data_root, 'NDVI4g/per_pix', year_range)
        path_type = 'dir'
        var_name = 'NDVI-origin'
        return self.__load_data(data_path, path_type), var_name

    def NDVI_anomaly_detrend(self,year_range=global_year_range):
        data_path = join(data_root, 'NDVI4g/per_pix_anomaly_detrend', year_range)
        path_type = 'dir'
        var_name = 'NDVI-anomaly_detrend'
        return self.__load_data(data_path, path_type), var_name

    def Temperature_origin(self,year_range=global_year_range):
        data_path = join(data_root, f'CRU_tmp/per_pix/{year_range}')
        path_type = 'dir'
        var_name = 'Temperature-origin'
        return self.__load_data(data_path, path_type), var_name

    def Temperature_origin_detrend(self,year_range=global_year_range):
        data_path = join(data_root, f'CRU_tmp/per_pix_detrend/{year_range}')
        path_type = 'dir'
        var_name = 'Temperature-origin_detrend'
        return self.__load_data(data_path, path_type), var_name

    def Temperature_anomaly(self,year_range=global_year_range):
        data_path = join(data_root, f'CRU_tmp/anomaly/{year_range}')
        path_type = 'dir'
        var_name = 'Temperature-anomaly'
        return self.__load_data(data_path, path_type), var_name

    def Temperature_anomaly_detrend(self,year_range=global_year_range):
        data_path = join(data_root, f'CRU_tmp/anomaly_detrend/{year_range}')
        path_type = 'dir'
        var_name = 'Temperature-anomaly_detrend'
        return self.__load_data(data_path, path_type), var_name

    def Precipitation_detrend(self,year_range=global_year_range):
        data_path = join(data_root, f'CRU_precip/detrend/{year_range}/precip.npy')
        path_type = 'file'
        var_name = 'Precipitation-detrend'
        return self.__load_data(data_path, path_type), var_name

    def Precipitation_origin(self,year_range=global_year_range):
        data_path = join(data_root, f'CRU_precip/per_pix/{year_range}')
        path_type = 'dir'
        var_name = 'Precipitation-origin'
        return self.__load_data(data_path, path_type), var_name

    def Precipitation_anomaly(self,year_range=global_year_range):
        data_path = join(data_root, f'CRU_precip/anomaly/{year_range}')
        path_type = 'dir'
        var_name = 'Precipitation-anomaly'
        return self.__load_data(data_path, path_type), var_name

    def Precipitation_anomaly_detrend(self,year_range=global_year_range):
        data_path = join(data_root, f'CRU_precip/anomaly_detrend/{year_range}')
        path_type = 'dir'
        var_name = 'Precipitation-anomaly_detrend'
        return self.__load_data(data_path, path_type), var_name

    def __load_data(self, data_path,path_type):
        if path_type == 'file':
            spatial_dict = T.load_npy(data_path)
        elif path_type == 'dir':
            spatial_dict = T.load_npy_dir(data_path)
        elif path_type == 'multi-files':
            spatial_dict = {}
            for f in T.listdir(data_path):
                print(f'loading {f}')
                key = f.split('.')[0]
                spatial_dict_i = T.load_npy(join(data_path, f))
                spatial_dict[key] = spatial_dict_i
        else:
            raise ValueError('path_type not recognized')
        return spatial_dict


def get_rs_rt_cols():
    post_n_list = [1, 2, 3, 4]
    cols = ['rt']
    for n in post_n_list:
        cols.append('rs_{}'.format(n))
    return cols

def year_range_str_to_list(year_range_str):
    year_range = year_range_str.split('-')
    year_range = list(range(int(year_range[0]), int(year_range[1]) + 1))
    return year_range