# coding=utf-8
import matplotlib.pyplot as plt
import semopy
import shap
import xgboost as xgb
from meta_info import *
import Rbeast as rb

result_root_this_script = join(results_root, 'attribution')


class MAT_Topt:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('MAT_Topt', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe/Dataframe.df')
        pass

    def run(self):
        # self.temperature_during_drought()
        # self.temperature_during_drought_delta()
        # self.build_df()
        # self.delta_temp_vs_compensation()
        # self.compensation_excerbation_delta()
        # self.compensation_excerbation_MAT_AI()
        self.compensation_excerbation_MAT_MAP()
        # self.pairplot()
        # self.hist_plot()
        # self.opt_mat_drought_temp_matrix()
        # self.delta_tif()
        pass


    def build_df(self):
        import analysis
        import statistic
        # Topt_f = join(analysis.Optimal_temperature().this_class_tif,'optimal_temperature/LT_Baseline_NT_origin_step_0.5_celsius_resample.tif')
        Topt_f = join(analysis.Optimal_temperature().this_class_tif,'optimal_temperature/TCSIF-optimal_temperature.tif')
        MAT_f = join(data_root,r"CRU_tmp\mat\mat_gs.tif")
        # MAT_f = join(data_root,r"CRU_tmp\max_annual_temperature\max_annual_temperature_gs.tif")
        method = 'max'
        Temp_hot_drought_f = join(self.this_class_tif,f'Temperature_during_drought/hot-drought_{method}_Temperature-origin_detrend.tif')
        Temp_normal_drought_f = join(self.this_class_tif,f'Temperature_during_drought/normal-drought_{method}_Temperature-origin_detrend.tif')

        compensation_excerbation_f = join(statistic.Compensation_Excerbation().this_class_tif,r"delta_hot_normal\drought_year_1.tif")

        Topt_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(Topt_f)
        MAT_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(MAT_f)
        Temp_hot_drought_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(Temp_hot_drought_f)
        Temp_normal_drought_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(Temp_normal_drought_f)
        compensation_excerbation_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(compensation_excerbation_f)

        all_dict = {}
        all_dict['Topt'] = Topt_spatial_dict
        all_dict['MAT'] = MAT_spatial_dict
        all_dict['Temp_hot_drought'] = Temp_hot_drought_spatial_dict
        all_dict['Temp_normal_drought'] = Temp_normal_drought_spatial_dict
        all_dict['compensation_excerbation'] = compensation_excerbation_spatial_dict

        df = T.spatial_dics_to_df(all_dict)
        df = df.dropna(how='any')

        outf = join(self.this_class_arr, 'dataframe/Dataframe_constant.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

        # for col in all_dict:
        #     spatial_dict = T.df_to_spatial_dic(df, col)
        #     arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        #     plt.figure()
        #     plt.imshow(arr,cmap='jet',interpolation='nearest',vmin=0,vmax=35)
        #     plt.colorbar()
        #     plt.title(col)
        # plt.show()

        pass

    def compensation_excerbation_delta(self):
        outdir = join(self.this_class_png,'compensation_excerbation_delta')
        T.mk_dir(outdir)
        import statistic
        compensation_excerbation_tif = join(statistic.Compensation_Excerbation().this_class_tif,'delta_hot_normal/drought_year_1.tif')
        compensation_excerbation_dict = DIC_and_TIF().spatial_tif_to_dic(compensation_excerbation_tif)
        Topt_MAT_delta_tif = join(self.this_class_tif,'mat_Topt_delta/Topt_MAT_delta.tif')
        Topt_MAT_delta_dict = DIC_and_TIF().spatial_tif_to_dic(Topt_MAT_delta_tif)

        spatial_dict_all = {
            'compensation_excerbation':compensation_excerbation_dict,
            'Topt_MAT_delta':Topt_MAT_delta_dict
        }
        df = T.spatial_dics_to_df(spatial_dict_all)
        df = statistic.Dataframe_func(df).df
        # df = pd.DataFrame()
        # df['compensation_excerbation_arr_flatten'] = compensation_excerbation_arr_flatten
        # df['Topt_MAT_delta_arr_flatten'] = Topt_MAT_delta_arr_flatten
        T.print_head_n(df)
        # exit()
        # df = df.dropna(how='any')
        bins_Topt_MAT_delta = np.linspace(-8,8,51)
        bins_AI = np.linspace(0,3,31)
        col_name = 'compensation_excerbation'
        df_group_Topt_MAT_delta, bins_list_str_Topt_MAT_delta = T.df_bin(df,'Topt_MAT_delta',bins_Topt_MAT_delta)
        plt.figure(figsize=(9*centimeter_factor,14*centimeter_factor))
        for name_Topt_MAT_delta, df_group_i_Topt_MAT_delta in df_group_Topt_MAT_delta:
            y_pos = name_Topt_MAT_delta[0].left
            df_group_AI, bins_list_str_AI = T.df_bin(df_group_i_Topt_MAT_delta,'aridity_index',bins_AI)
            for name_AI, df_group_i_AI in df_group_AI:
                x_pos = name_AI[0].left
                vals = df_group_i_AI[col_name].tolist()
                if len(vals) == 0:
                    continue
                if T.is_all_nan(vals):
                    continue
                mean = np.nanmean(vals)
                plt.scatter(x_pos,y_pos,s=13,c=mean,vmin=-0.5,vmax=0.5,cmap='RdBu',marker='s',linewidths=0)
        plt.colorbar()
        outf = join(outdir,'compensation_excerbation_delta.pdf')
        # plt.savefig(outf)
        plt.show()

        # plt.hist(df['Topt_MAT_delta'], bins=100, range=(-8, 8), zorder=-99,color='gray',alpha=0.5)
        # # plt.hist(df['Topt_MAT_delta_arr_flatten'], bins=100, zorder=-99,color='gray',alpha=0.5)
        # plt.xlabel('MAT - Topt')
        # plt.ylabel('Compensation Excerbation')
        # plt.twinx()
        # plt.plot(x_list, y_list, c='r')
        # plt.fill_between(x_list, np.array(y_list) - np.array(err_list), np.array(y_list) + np.array(err_list),
        #                  alpha=0.3)
        #
        # plt.show()
        # exit()
        pass

    def compensation_excerbation_MAT_AI(self):
        outdir = join(self.this_class_png,'compensation_excerbation_MAT_AI')
        T.mk_dir(outdir)
        import statistic
        compensation_excerbation_tif = join(statistic.Compensation_Excerbation().this_class_tif,'delta_hot_normal/drought_year_1.tif')
        compensation_excerbation_dict = DIC_and_TIF().spatial_tif_to_dic(compensation_excerbation_tif)
        # Topt_MAT_delta_tif = join(self.this_class_tif,'mat_Topt_delta/Topt_MAT_delta.tif')
        # Topt_MAT_delta_dict = DIC_and_TIF().spatial_tif_to_dic(Topt_MAT_delta_tif)

        MAT_tif = join(data_root,r'CRU_tmp\mat','mat_gs.tif')
        MAT_dict = DIC_and_TIF().spatial_tif_to_dic(MAT_tif)
        spatial_dict_all = {
            'compensation_excerbation':compensation_excerbation_dict,
            'MAT':MAT_dict
        }
        df = T.spatial_dics_to_df(spatial_dict_all)
        df = statistic.Dataframe_func(df).df
        # df = pd.DataFrame()
        # df['compensation_excerbation_arr_flatten'] = compensation_excerbation_arr_flatten
        # df['Topt_MAT_delta_arr_flatten'] = Topt_MAT_delta_arr_flatten
        T.print_head_n(df)
        # exit()
        # df = df.dropna(how='any')
        bins_Topt_MAT_delta = np.linspace(-5,35,51)
        bins_AI = np.linspace(0,3,31)
        col_name = 'compensation_excerbation'
        df_group_Topt_MAT_delta, bins_list_str_Topt_MAT_delta = T.df_bin(df,'MAT',bins_Topt_MAT_delta)
        plt.figure(figsize=(9*centimeter_factor,14*centimeter_factor))
        for name_Topt_MAT_delta, df_group_i_Topt_MAT_delta in df_group_Topt_MAT_delta:
            y_pos = name_Topt_MAT_delta[0].left
            df_group_AI, bins_list_str_AI = T.df_bin(df_group_i_Topt_MAT_delta,'aridity_index',bins_AI)
            for name_AI, df_group_i_AI in df_group_AI:
                x_pos = name_AI[0].left
                vals = df_group_i_AI[col_name].tolist()
                if len(vals) == 0:
                    continue
                if T.is_all_nan(vals):
                    continue
                mean = np.nanmean(vals)
                plt.scatter(x_pos,y_pos,s=13,c=mean,vmin=-0.5,vmax=0.5,cmap='RdBu',marker='s',linewidths=0)
        plt.colorbar()
        plt.xlabel('Aridity Index')
        plt.ylabel('MAT')
        outf = join(outdir,'compensation_excerbation_MAT_AI.pdf')
        # plt.savefig(outf)
        plt.show()

        # plt.hist(df['Topt_MAT_delta'], bins=100, range=(-8, 8), zorder=-99,color='gray',alpha=0.5)
        # # plt.hist(df['Topt_MAT_delta_arr_flatten'], bins=100, zorder=-99,color='gray',alpha=0.5)
        # plt.xlabel('MAT - Topt')
        # plt.ylabel('Compensation Excerbation')
        # plt.twinx()
        # plt.plot(x_list, y_list, c='r')
        # plt.fill_between(x_list, np.array(y_list) - np.array(err_list), np.array(y_list) + np.array(err_list),
        #                  alpha=0.3)
        #
        # plt.show()
        # exit()
        pass

    def compensation_excerbation_MAT_MAP(self):
        outdir = join(self.this_class_png,'compensation_excerbation_MAT_MAP')
        T.mk_dir(outdir)
        import statistic
        compensation_excerbation_tif = join(statistic.Compensation_Excerbation().this_class_tif,'delta_hot_normal/drought_year_1.tif')
        compensation_excerbation_dict = DIC_and_TIF().spatial_tif_to_dic(compensation_excerbation_tif)
        # Topt_MAT_delta_tif = join(self.this_class_tif,'mat_Topt_delta/Topt_MAT_delta.tif')
        # Topt_MAT_delta_dict = DIC_and_TIF().spatial_tif_to_dic(Topt_MAT_delta_tif)

        MAT_tif = join(data_root,r'CRU_tmp\mat','mat_gs.tif')
        MAT_dict = DIC_and_TIF().spatial_tif_to_dic(MAT_tif)

        MAP_tif = join(data_root,r'CRU_precip\map','map.tif')
        MAP_dict = DIC_and_TIF().spatial_tif_to_dic(MAP_tif)
        spatial_dict_all = {
            'compensation_excerbation':compensation_excerbation_dict,
            'MAT':MAT_dict,
            'MAP':MAP_dict
        }
        df = T.spatial_dics_to_df(spatial_dict_all)
        df = df.dropna(how='any')
        map = df['MAP'].tolist()
        # plt.hist(map, bins=100, range=(0, 1500), zorder=-99, color='gray', alpha=0.5)
        # plt.show()
        T.print_head_n(df)
        # exit()
        # df = df.dropna(how='any')
        bins_Topt_MAT_delta = np.arange(-5,36,1)
        bins_MAP = np.arange(0,4001,100)
        col_name = 'compensation_excerbation'
        df_group_Topt_MAT_delta, bins_list_str_Topt_MAT_delta = T.df_bin(df,'MAT',bins_Topt_MAT_delta)
        # plt.figure(figsize=(10*centimeter_factor,11*centimeter_factor))
        # plt.figure(figsize=(10*centimeter_factor,11*centimeter_factor))
        # plt.figure(figsize=(16*centimeter_factor,18*centimeter_factor))
        for name_Topt_MAT_delta, df_group_i_Topt_MAT_delta in df_group_Topt_MAT_delta:
            y_pos = name_Topt_MAT_delta[0].left
            df_group_MAP, bins_list_str_MAP = T.df_bin(df_group_i_Topt_MAT_delta,'MAP',bins_MAP)
            for name_MAP, df_group_i_MAP in df_group_MAP:
                x_pos = name_MAP[0].left
                vals = df_group_i_MAP[col_name].tolist()
                # if len(vals) < 10:
                #     continue
                if T.is_all_nan(vals):
                    continue
                mean = np.nanmean(vals)
                plt.scatter(y_pos,x_pos,s=40,c=mean,vmin=-0.5,vmax=0.5,cmap='RdBu',marker='s',linewidths=0)
                # plt.scatter(x_pos,y_pos,s=13,c=mean,vmin=-0.01,vmax=0.01,cmap='RdBu',marker='s',linewidths=0)
        plt.colorbar()
        plt.xlabel('MAP')
        plt.ylabel('MAT')
        outf = join(outdir,'compensation_excerbation_MAT_MAP.pdf')
        # plt.savefig(outf)
        plt.show()

        # plt.hist(df['Topt_MAT_delta'], bins=100, range=(-8, 8), zorder=-99,color='gray',alpha=0.5)
        # # plt.hist(df['Topt_MAT_delta_arr_flatten'], bins=100, zorder=-99,color='gray',alpha=0.5)
        # plt.xlabel('MAT - Topt')
        # plt.ylabel('Compensation Excerbation')
        # plt.twinx()
        # plt.plot(x_list, y_list, c='r')
        # plt.fill_between(x_list, np.array(y_list) - np.array(err_list), np.array(y_list) + np.array(err_list),
        #                  alpha=0.3)
        #
        # plt.show()
        # exit()
        pass

    def temperature_during_drought(self):
        outdir = join(self.this_class_tif,'temperature_during_drought')
        T.mk_dir(outdir)
        dff = SEM().dff
        df = T.load_df(dff)
        drt_list = ['hot-drought','normal-drought']
        method = 'mean'
        # method = 'max'

        for drt in drt_list:
            df_drt = df[df['drought_type'] == drt]
            # df = self.add_variables(df,Load_Data().Temperature_origin,method=method)
            df_drt,key_name = self.add_variables(df_drt,Load_Data().Temperature_origin_detrend,method=method)
            T.print_head_n(df_drt)
            # exit()
            df_pix_dict = T.df_groupby(df_drt, 'pix')
            Temperature_origin_mean_dict = {}
            for pix in df_pix_dict:
                df_pix = df_pix_dict[pix]
                df_pix = T.df_drop_duplicates(df_pix, 'drought_year')
                # Temperature_origin = df_pix['max_Temperature-origin'].tolist()
                Temperature_origin = df_pix[key_name].tolist()
                Temperature_origin_mean = np.nanmean(Temperature_origin)
                Temperature_origin_mean_dict[pix] = Temperature_origin_mean
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(Temperature_origin_mean_dict)
            # outf = join(outdir,f'{drt}_max_Temperature.tif')
            outf = join(outdir,f'{drt}_{key_name}.tif')
            DIC_and_TIF().arr_to_tif(arr,outf)

    def temperature_during_drought_delta(self):
        fdir = join(self.this_class_tif,'temperature_during_drought')
        outdir = join(self.this_class_tif,'temperature_during_drought_delta')
        T.mk_dir(outdir)
        method = 'mean'
        # method = 'max'
        hot_drought_tif = join(fdir,f'hot-drought_{method}_Temperature-origin_detrend.tif')
        normal_drought_tif = join(fdir,f'normal-drought_{method}_Temperature-origin_detrend.tif')
        hot_drought_arr = DIC_and_TIF().spatial_tif_to_arr(hot_drought_tif)
        normal_drought_arr = DIC_and_TIF().spatial_tif_to_arr(normal_drought_tif)
        arr = hot_drought_arr - normal_drought_arr
        # arr_flatten = arr.flatten()
        # plt.hist(arr_flatten,bins=100)
        # plt.figure()
        # plt.imshow(arr,cmap='RdBu_r',vmin=-5,vmax=5,interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        # outf = join(outdir,'delta.tif')
        # DIC_and_TIF().arr_to_tif(arr,outf)
        return arr

    def add_variables(self,df,data_obj,method='mean'):
        # data_dict,var_name,valid_range = Load_Data().NDVI_anomaly_detrend()
        data_dict,var_name,valid_range = data_obj()

        vals_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            year = row['drought_year']
            if not pix in data_dict:
                vals_list.append(np.nan)
                continue
            vals = data_dict[pix]
            vals = np.array(vals)
            if np.nanstd(vals) == 0:
                vals_list.append(np.nan)
                continue

            vals[vals<valid_range[0]] = np.nan
            vals[vals>valid_range[1]] = np.nan
            # vals_reshape = np.reshape(vals,(-1,12))
            # vals_gs = T.monthly_vals_to_annual_val(vals,grow_season=global_gs,method='max')
            vals_gs = T.monthly_vals_to_annual_val(vals,grow_season=global_gs,method=method)
            val_drought_year = vals_gs[year-global_start_year]
            vals_list.append(val_drought_year)

        # df['max_'+var_name] = vals_list
        df[f'{method}_'+var_name] = vals_list
        return df,f'{method}_'+var_name

    def pairplot(self):

        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        compensation_excerbation_class = []
        for i,row in df.iterrows():
            compensation_excerbation = row['compensation_excerbation']
            if compensation_excerbation < 0:
                compensation_excerbation_class.append('excerbation')
            elif compensation_excerbation > 0:
                compensation_excerbation_class.append('compensation')
            else:
                compensation_excerbation_class.append('neutral')
        df['compensation_excerbation_class'] = compensation_excerbation_class
        sns.pairplot(df,vars=['MAT','Temp_hot_drought','Temp_normal_drought','Topt'],hue='compensation_excerbation_class',)
        # plt.xlim(0, 40)
        # plt.ylim(0, 40)
        plt.show()

        pass

    def hist_plot(self):
        dff = self.dff
        df = T.load_df(dff)
        compensation_excerbation_class = []
        for i, row in df.iterrows():
            compensation_excerbation = row['compensation_excerbation']
            if compensation_excerbation < 0:
                compensation_excerbation_class.append('excerbation')
            elif compensation_excerbation > 0:
                compensation_excerbation_class.append('compensation')
            else:
                compensation_excerbation_class.append('neutral')
        df['compensation_excerbation_class'] = compensation_excerbation_class
        compensation_excerbation_class_list = ['compensation','excerbation']
        vars_list = ['MAT', 'Temp_hot_drought', 'Temp_normal_drought', 'Topt']
        color_list = ['r', 'g', 'b', 'k']
        flag = 0
        for var in vars_list:
            plt.subplot(2, 2, flag + 1)
            for c_e in compensation_excerbation_class_list:
                if c_e == 'excerbation':
                    alpha = 1
                else:
                    alpha = 0.3
                # plt.figure()
                df_ce = df[df['compensation_excerbation_class'] == c_e]
                box_list = []

                vals = df_ce[var].tolist()
                vals = np.array(vals)
                box_list.append(vals)
                x,y = Plot().plot_hist_smooth(vals, bins=51,alpha=0,interpolate_window=11,range=(0,40))
                plt.plot(x,y,label=c_e,color=color_list[flag],lw=3,alpha=alpha)
                # plt.twinx()
                plt.title(var)
            flag += 1

            plt.legend()
            # plt.xlim(0, 40)
            # plt.ylim(0, 40)
            # plt.tight_layout()
        plt.show()

        pass

    def delta_temp_vs_compensation(self):
        dff = join(self.this_class_arr, 'dataframe/Dataframe_constant.df')
        df = T.load_df(dff)
        T.print_head_n(df)
        delta_hot_normal_temp = df['Temp_hot_drought'] - df['Temp_normal_drought']
        df['delta_hot_normal_temp'] = delta_hot_normal_temp
        x = df['delta_hot_normal_temp'].tolist()
        y = df['compensation_excerbation'].tolist()
        KDE_plot().plot_scatter(x,y)
        plt.xlabel('delta_hot_normal_temp')
        plt.ylabel('compensation_excerbation')
        # plt.scatter(x,y)
        plt.show()

        pass

    def opt_mat_drought_temp_matrix(self):
        outdir = join(self.this_class_png, 'opt_mat_drought_temp_matrix')
        T.mkdir(outdir,force=True)
        # import statistic
        # import analysis
        # dff = statistic.Drought_timing().dff
        # df = T.load_df(dff)
        # df = df.drop(columns=['optimal_temp'])
        # method = 'max'
        # # method = 'mean'
        # df,var_name = self.add_variables(df,Load_Data().Temperature_origin_detrend,method=method)
        # # T.print_head_n(df)
        # # exit()
        # Topt_f = join(analysis.Optimal_temperature().this_class_tif,
        #               'optimal_temperature/TCSIF-optimal_temperature.tif')
        # # MAT_f = join(data_root,r"CRU_tmp\mat\mat_gs.tif")
        # MAT_f = join(data_root, r"CRU_tmp\max_annual_temperature\max_annual_temperature_gs.tif")
        #
        # # df = self.add_variables()
        # Topt_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(Topt_f)
        # MAT_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(MAT_f)
        # df = T.add_spatial_dic_to_df(df, Topt_spatial_dict, 'Topt')
        # df = T.add_spatial_dic_to_df(df, MAT_spatial_dict, 'MAT')
        # T.print_head_n(df)
        #
        # delta_mat_topt = df['MAT'] - df['Topt']
        # delta_temp_hot_drought = df[var_name] - df['Topt']
        #
        # df['delta_mat_topt'] = delta_mat_topt
        # df['delta_temp_hot_drought_topt'] = delta_temp_hot_drought
        # T.save_df(df, self.dff)
        # T.df_to_excel(df, self.dff)
        season_index_dict = {
            'spring':[0,1],
            'summer':[2,3],
            'autumn':[4,5],
        }
        df = T.load_df(self.dff)
        delta_mat_topt = df['delta_mat_topt'].tolist()
        delta_temp_hot_drought = df['delta_temp_hot_drought_topt'].tolist()
        plt.hist(delta_mat_topt,bins=100)
        plt.title('delta_mat_topt')
        plt.figure()
        plt.hist(delta_temp_hot_drought,bins=100)
        plt.title('delta_temp_hot_drought_topt')
        # plt.show()

        result_dict = {
            'spring':[],
            'summer':[],
            'autumn':[],
        }
        df = df[df['drought_type'] == 'hot-drought']
        # df = df[df['drought_type'] == 'normal-drought']

        for i,row in tqdm(df.iterrows(),total=len(df)):
            drought_season = row['drought_season']
            NDVI_progress = row['NDVI_progress'].tolist()
            NDVI_progress_reshape = np.reshape(NDVI_progress, (-1, 6))
            drought_year_NDVI = NDVI_progress_reshape[1]
            for season in season_index_dict:
                vals_season = drought_year_NDVI[season_index_dict[season]]
                vals_season_mean = np.nanmean(vals_season)
                result_dict[season].append(vals_season_mean)
        for season in result_dict:
            df[f'{season}_NDVI'] = result_dict[season]


        ELI_class = 'Energy-Limited'
        # season = 'spring'
        # season = 'summer'
        # df = df[df['drought_season'] == 'spring']
        # df = df[df['drought_season'] == season]
        # for ELI_class in ['Energy-Limited','Water-Limited']:
        # df = df[df['ELI_class'] == ELI_class]
        for season in season_index_dict:
            # df_season = df_ELI[df_ELI['drought_season'] == season]
            df_season = df[df['drought_season'] == season]
            delta_mat_topt = df_season['delta_mat_topt'].tolist()
            delta_temp_hot_drought = df_season['delta_temp_hot_drought_topt'].tolist()
            delta_mat_topt_bins = np.linspace(-5, 10, 31)
            delta_temp_hot_drought_bins = np.linspace(-5, 10, 31)
            df_group_mat_topt, bins_list_strmat_topt = T.df_bin(df_season, 'delta_mat_topt', delta_mat_topt_bins)
            plt.figure()
            for name_i_mat_topt,df_group_i_mat_topt in df_group_mat_topt:
                y_name = name_i_mat_topt[0].left
                # print(y_name)
                df_group_i_hot_topt, bins_list_strhot_topt = T.df_bin(df_group_i_mat_topt, 'delta_temp_hot_drought_topt',delta_temp_hot_drought_bins)
                for name_i_hot_topt,df_group_i_hot_topt in df_group_i_hot_topt:
                    x_name = name_i_hot_topt[0].left
                    rt_list = df_group_i_hot_topt[f'{season}_NDVI'].tolist()
                    # rt_list = df_group_i_hot_topt['rt'].tolist()
                    rt_mean = np.nanmean(rt_list)
                    plt.scatter(x_name,y_name,c=rt_mean,alpha=1,vmin=-1,vmax=1,cmap='RdBu',marker='s')
                    # plt.scatter(x_name,y_name,c=rt_mean,alpha=1,vmin=0.95,vmax=1.05,cmap='RdBu',marker='s')
            plt.colorbar()
            plt.axis('equal')
            plt.xlabel('delta_temp_hot_drought_topt')
            plt.ylabel('delta_mat_topt')
            # plt.title(f'{season}_{ELI_class}')
            # plt.title(f'{season} {ELI_class}')
            plt.title(f'{season}')
            outf = join(outdir,f'{season}.pdf')
            plt.savefig(outf)
            plt.close()
        # plt.show()

        pass


    def delta_tif(self):
        outdir = join(self.this_class_tif,'delta_tif')
        T.mk_dir(outdir)
        dff = self.dff
        df = T.load_df(dff)
        # var_name = 'delta_temp_hot_drought_topt'
        var_name = 'delta_mat_topt'
        df_pix_group = T.df_groupby(df,'pix')
        spatial_dict = {}
        for pix in df_pix_group:
            df_pix = df_pix_group[pix]
            vals = df_pix[var_name].tolist()[0]
            spatial_dict[pix] = vals
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        outf = join(outdir,f'{var_name}.tif')
        DIC_and_TIF().arr_to_tif(arr,outf)
        # plt.imshow(arr,cmap='RdBu_r',vmin=-5,vmax=5,interpolation='nearest')
        # plt.colorbar()
        # plt.title(var_name)
        # plt.show()
        pass


class MAT_Topt1:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('MAT_Topt1', result_root_this_script, mode=2)

    def run(self):
        self.delta_tif()
        pass

    def delta_tif(self):
        import analysis
        import statistic
        Topt_f = join(analysis.Optimal_temperature().this_class_tif,r"optimal_temperature\LT_Baseline_NT_origin_step_0.5_celsius_resample.tif")
        # Topt_f = join(analysis.Optimal_temperature().this_class_tif,r"optimal_temperature\TCSIF-origin_step_0.5_celsius_Max-Temperature-origin.tif")
        # compensation_excerbation_f = join(statistic.Compensation_Excerbation().this_class_tif,r"delta_hot_normal\drought_year_1.tif")
        delta_tif = join(statistic.Drought_timing().this_class_tif,r"delta\delta.tif")
        # mat_tif = join(data_root,r"CRU_tmp\max_annual_temperature\max_annual_temperature_gs.tif")
        # mat_tif = join(data_root,r"CRU_tmp\mat\mat_gs.tif")
        mat_tif = join(data_root,r"CRU_tmx\mat\mat_gs.tif")
        # mat_tif = join(data_root,r"CRU_tmx\max_annual_temperature\max_annual_temperature.tif")

        tif_dict = {
            'Topt':Topt_f,
            # 'compensation_excerbation':compensation_excerbation_f,
            'delta':delta_tif,
            'mat':mat_tif
        }
        df = self.tifs_to_df(tif_dict)
        df = statistic.Dataframe_func(df).df
        T.print_head_n(df)
        # exit()
        df['T_delta'] = df['mat'] - df['Topt']
        T_delta = df['T_delta'].tolist()
        # plt.hist(T_delta,bins=100)
        # plt.show()
        AI_bin = np.linspace(0,3,31)
        T_delta_bins = np.linspace(-30,30,51)
        delta = df['delta'].tolist()
        # Topt = df['Topt'].tolist()
        T_delta = df['T_delta'].tolist()

        df_group_AI, bins_list_str = T.df_bin(df,'aridity_index',AI_bin)
        for name_AI,df_group_AI_i in df_group_AI:
            x = name_AI[0].left
            df_group_Tdelta, bins_list_str = T.df_bin(df_group_AI_i,'T_delta',T_delta_bins)
            for name_Tdelta,df_group_Tdelta_i in df_group_Tdelta:
                y = name_Tdelta[0].left
                vals = df_group_Tdelta_i['delta'].tolist()
                if T.is_all_nan(vals):
                    continue
                # if len(vals) < 10:
                #     continue
                vals_mean = np.nanmean(vals)
                # vals_mean = len(vals)
                plt.scatter(x,y,c=vals_mean,vmin=-.5,vmax=.5,cmap='RdBu',marker='s')
                # plt.scatter(x,y,c=vals_mean,vmin=0,vmax=1000,cmap='RdBu',marker='s')
        plt.colorbar()
        plt.xlabel('aridity_index')
        plt.ylabel('MAT - Topt')
        plt.show()


        pass

    def tifs_to_df(self,tif_dict):
        tif_template = ''
        for key in tif_dict:
            tif_template = tif_dict[key]
            break
        D_template = DIC_and_TIF(tif_template=tif_template)
        spatial_dicts = {}
        for key in tif_dict:
            spatial_dict = D_template.spatial_tif_to_dic(tif_dict[key])
            spatial_dicts[key] = spatial_dict
        df = T.spatial_dics_to_df(spatial_dicts)
        return df

class Attribution_Dataframe:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Attribution_Dataframe', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe', 'dataframe.df')
        pass

    def run(self):
        # self.copy_df()
        df = self.__gen_df_init()

        # df = self.add_NDVI(df)
        # df = self.add_NDVI_with_trend(df)
        # df = self.add_SOS(df)
        # df = self.add_VPD(df)
        # df = self.add_Temperature_quantile(df)
        # df = self.add_max_T(df)
        # df = self.add_mean_T(df)
        # df = self.add_Topt(df)
        # df = self.add_PAR(df)
        # df = self.add_srad(df)
        # df = self.add_T_anomaly(df)
        # df = self.add_SMRoot_anomaly(df)
        # df = self.add_SMRoot_origin(df)
        # df = self.add_Soil_SILT(df)
        # df = self.add_Soil_sand(df)
        # df = self.add_detrend_T_anomaly(df)
        # df = self.add_delta_Topt_T(df)
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)
        # self.check_variable(df)
        # self.check_Topt_vs_ndvi(df)
        # self.plot_variables(df)
        # self.print_drought_events_numbers(df)

    def __gen_df_init(self):
        if not os.path.isfile(self.dff):
            df = pd.DataFrame()
            T.save_df(df,self.dff)
            return df
        else:
            df,dff = self.__load_df()
            return df

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print('len(df):',len(df))
        return df,dff

    def variables_info(self):
        x_list = [
            'SOS',
            'VPD-anomaly',
            # 'Temperature-anomaly_detrend',
            # 'Temperature_quantile',
            'Temperature-anomaly',
            # 'FAPAR-anomaly_detrend',
            'GLEAM-SMRoot-anomaly_detrend',
            'Radiation-anomaly',
            # 'delta_Topt_T',
            'drought_mon',
            'sand',
        ]
        y = 'NDVI-anomaly_detrend'
        return x_list,y

    def variables_threshold(self):
        variables_threshold_dict = {
            'SOS':(-20,20),
            'VPD-anomaly':(-3,3),
            'Temperature-anomaly_detrend':(-3,3),
            'Temperature-anomaly':(-3,3),
            'Temperature_quantile':(0,100),
            'FAPAR-anomaly_detrend':(-3,3),
            'Radiation-anomaly':(-3,3),
            'delta_Topt_T':(-10,10),
            'drought_mon':(5,10),
            'NDVI-anomaly_detrend':(-3,3),
            'NDVI-anomaly_with_trend':(-3,3),
            'GLEAM-SMRoot-anomaly_detrend':(-3,3),
            'SILT':(0,100),
            'sand':(0,1000),

        }
        return variables_threshold_dict

    def clean_df(self,df):
        delta_threshold_dict = self.variables_threshold()
        x_variables,y_variable = self.variables_info()
        # print(x_variables)
        # print(delta_threshold_dict)
        for key in x_variables:
            left,right = delta_threshold_dict[key]
            # print(key,(left,right))
            df = df[df[key]>=left]
            df = df[df[key]<=right]
        y_left,y_right = delta_threshold_dict[y_variable]
        df = df[df[y_variable]>=y_left]
        df = df[df[y_variable]<=y_right]

        return df

    def copy_df(self):
        import statistic
        T.mkdir(join(self.this_class_arr, 'dataframe'))
        if isfile(self.dff):
            print('already exists: ', self.dff)
            print('press enter to overwrite')
            pause()
            pause()
            pause()
        T.mkdir(join(self.this_class_arr, 'dataframe'))
        dff = join(statistic.Drought_timing().this_class_arr, r"Drought_timing.df")
        df = T.load_df(dff)
        T.save_df(df,self.dff)
        T.df_to_excel(df, self.dff)

    def add_SOS(self,df):
        import analysis
        pheno_str = 'early_start'
        fpath = join(analysis.Phenology().this_class_arr, 'phenology_df/phenology_df.df')
        phenology_df = T.load_df(fpath)
        cols = list(phenology_df.columns)
        print(cols)
        pheno_spatial_dict = {}
        for i, row in phenology_df.iterrows():
            pix = row['pix']
            early_start = row[pheno_str]
            early_start_dict = dict(early_start)
            phenology_anomaly_dict = self.phenology_anomaly_dict(early_start_dict)
            pheno_spatial_dict[pix] = phenology_anomaly_dict
        pheno_val_list = []
        for i,row in df.iterrows():
            pix = row['pix']
            year = row['drought_year']
            if not pix in pheno_spatial_dict:
                pheno_val_list.append(np.nan)
                continue
            if not year in pheno_spatial_dict[pix]:
                pheno_val_list.append(np.nan)
                continue
            pheno_val = pheno_spatial_dict[pix][year]
            pheno_val_list.append(pheno_val)
        df['SOS'] = pheno_val_list
        return df

    def add_Topt(self,df):
        import analysis
        fpath = join(analysis.Optimal_temperature().this_class_tif,'optimal_temperature/TCSIF-optimal_temperature.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'optimal_temp')
        return df

    def phenology_anomaly_dict(self,phenology_dict):
        vals = list(phenology_dict.values())
        mean = np.nanmean(vals)
        phenology_anomaly_dict = {}
        for year in phenology_dict:
            val = phenology_dict[year]
            anomaly = val - mean
            phenology_anomaly_dict[year] = anomaly
        return phenology_anomaly_dict

    def add_VPD(self,df):
        # df = Load_dataframe()
        vals_spatial_dict,data_name,valid_range = Load_Data().VPD_anomaly()
        # print(data_name)
        # exit()
        year_list = global_year_range_list
        gs = global_gs
        vals_drought_year_mean_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            # print(drought_year)
            # exit()
            vals = vals_spatial_dict[pix]
            vals = np.array(vals,dtype=float)
            vals[vals>valid_range[1]] = np.nan
            vals[vals<valid_range[0]] = np.nan
            # vals[vals<0] = np.nan
            vals_gs = T.monthly_vals_to_annual_val(vals,gs,method='array')
            vals_gs_dict = T.dict_zip(year_list,vals_gs)
            vals_drought_year = vals_gs_dict[drought_year]
            vals_drought_year_mean = np.nanmean(vals_drought_year)
            vals_drought_year_mean_list.append(vals_drought_year_mean)
        df[f'{data_name}'] = vals_drought_year_mean_list
        return df

    def add_Temperature_quantile(self,df):
        # df = Load_dataframe()
        vals_spatial_dict,data_name,valid_range = Load_Data().Temperature_quantile()
        # print(data_name)
        # exit()
        year_list = global_year_range_list
        gs = global_gs
        vals_drought_year_mean_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            # print(drought_year)
            # exit()
            vals = vals_spatial_dict[pix]
            vals = np.array(vals,dtype=float)
            vals[vals>valid_range[1]] = np.nan
            vals[vals<valid_range[0]] = np.nan
            # vals[vals<0] = np.nan
            vals_gs = T.monthly_vals_to_annual_val(vals,gs,method='array')
            vals_gs_dict = T.dict_zip(year_list,vals_gs)
            vals_drought_year = vals_gs_dict[drought_year]
            vals_drought_year_mean = np.nanmean(vals_drought_year)
            vals_drought_year_mean_list.append(vals_drought_year_mean)
        df[f'{data_name}'] = vals_drought_year_mean_list
        return df

    def add_max_T(self,df):
        # df = Load_dataframe()
        vals_spatial_dict,data_name,valid_range = Load_Data().Temperature_origin()
        # print(data_name)
        # exit()
        year_list = global_year_range_list
        gs = global_gs
        vals_drought_year_mean_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            # print(drought_year)
            # exit()
            vals = vals_spatial_dict[pix]
            vals = np.array(vals,dtype=float)
            vals[vals>valid_range[1]] = np.nan
            vals[vals<valid_range[0]] = np.nan
            # vals[vals<0] = np.nan
            vals_gs = T.monthly_vals_to_annual_val(vals,gs,method='array')
            vals_gs_dict = T.dict_zip(year_list,vals_gs)
            vals_drought_year = vals_gs_dict[drought_year]
            vals_drought_year_mean = np.nanmax(vals_drought_year)
            vals_drought_year_mean_list.append(vals_drought_year_mean)
        df[f'max_{data_name}'] = vals_drought_year_mean_list
        return df

    def add_mean_T(self,df):
        # df = Load_dataframe()
        vals_spatial_dict,data_name,valid_range = Load_Data().Temperature_origin()
        # print(data_name)
        # exit()
        year_list = global_year_range_list
        gs = global_gs
        vals_drought_year_mean_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            # print(drought_year)
            # exit()
            vals = vals_spatial_dict[pix]
            vals = np.array(vals,dtype=float)
            vals[vals>valid_range[1]] = np.nan
            vals[vals<valid_range[0]] = np.nan
            # vals[vals<0] = np.nan
            vals_gs = T.monthly_vals_to_annual_val(vals,gs,method='array')
            vals_gs_dict = T.dict_zip(year_list,vals_gs)
            vals_drought_year = vals_gs_dict[drought_year]
            vals_drought_year_mean = np.nanmean(vals_drought_year)
            vals_drought_year_mean_list.append(vals_drought_year_mean)
        df[f'mean_{data_name}'] = vals_drought_year_mean_list
        return df

    def add_PAR(self,df):
        # df = Load_dataframe()
        vals_spatial_dict,data_name,valid_range = Load_Data().FAPAR_anomaly_detrend()
        # print(data_name)
        # exit()
        year_list = global_year_range_list
        gs = global_gs
        vals_drought_year_mean_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            # print(drought_year)
            # exit()
            vals = vals_spatial_dict[pix]
            vals = np.array(vals,dtype=float)
            vals[vals>valid_range[1]] = np.nan
            vals[vals<valid_range[0]] = np.nan
            # vals[vals<0] = np.nan
            vals_gs = T.monthly_vals_to_annual_val(vals,gs,method='array')
            vals_gs_dict = T.dict_zip(year_list,vals_gs)
            vals_drought_year = vals_gs_dict[drought_year]
            vals_drought_year_mean = np.nanmean(vals_drought_year)
            vals_drought_year_mean_list.append(vals_drought_year_mean)
        df[f'{data_name}'] = vals_drought_year_mean_list
        return df

    def add_srad(self,df):
        # df = Load_dataframe()
        vals_spatial_dict,data_name,valid_range = Load_Data().Srad_anomaly()
        # print(data_name)
        # exit()
        year_list = global_year_range_list
        gs = global_gs
        vals_drought_year_mean_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            # print(drought_year)
            # exit()
            vals = vals_spatial_dict[pix]
            vals = np.array(vals,dtype=float)
            vals[vals>valid_range[1]] = np.nan
            vals[vals<valid_range[0]] = np.nan
            # vals[vals<0] = np.nan
            vals_gs = T.monthly_vals_to_annual_val(vals,gs,method='array')
            vals_gs_dict = T.dict_zip(year_list,vals_gs)
            vals_drought_year = vals_gs_dict[drought_year]
            vals_drought_year_mean = np.nanmean(vals_drought_year)
            vals_drought_year_mean_list.append(vals_drought_year_mean)
        df[f'{data_name}'] = vals_drought_year_mean_list
        return df

    def add_T_anomaly(self,df):
        # df = Load_dataframe()
        vals_spatial_dict,data_name,valid_range = Load_Data().Temperature_anomaly()
        # print(data_name)
        # exit()
        year_list = global_year_range_list
        gs = global_gs
        vals_drought_year_mean_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            # print(drought_year)
            # exit()
            vals = vals_spatial_dict[pix]
            vals = np.array(vals,dtype=float)
            vals[vals>valid_range[1]] = np.nan
            vals[vals<valid_range[0]] = np.nan
            # vals[vals<0] = np.nan
            vals_gs = T.monthly_vals_to_annual_val(vals,gs,method='array')
            vals_gs_dict = T.dict_zip(year_list,vals_gs)
            vals_drought_year = vals_gs_dict[drought_year]
            vals_drought_year_mean = np.nanmean(vals_drought_year)
            vals_drought_year_mean_list.append(vals_drought_year_mean)
        df[f'{data_name}'] = vals_drought_year_mean_list
        return df

    def add_SMRoot_anomaly(self,df):
        # df = Load_dataframe()
        vals_mean_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            NDVI_progress = row['GLEAM-SMRoot-anomaly_detrend_progress'].tolist()
            # print(NDVI_progress)
            try:
                NDVI_progress_reshape = np.reshape(NDVI_progress, (-1, 6))
            except:
                vals_mean_list.append(np.nan)
                continue
            if len(NDVI_progress_reshape) != 6:
                vals_mean_list.append(np.nan)
                continue
            drought_year_NDVI = NDVI_progress_reshape[1]
            drought_year_NDVI_mean = np.nanmean(drought_year_NDVI)
            vals_mean_list.append(drought_year_NDVI_mean)
        df['GLEAM-SMRoot-anomaly_detrend'] = vals_mean_list
        return df

    def add_SMRoot_origin(self,df):
        # df = Load_dataframe()
        vals_spatial_dict,data_name,valid_range = Load_Data().GLEAM_SMRoot_origin()
        # print(data_name)
        # exit()
        year_list = global_year_range_list
        gs = global_gs
        vals_drought_year_mean_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            # print(drought_year)
            # exit()
            vals = vals_spatial_dict[pix]
            vals = np.array(vals,dtype=float)
            vals[vals>valid_range[1]] = np.nan
            vals[vals<valid_range[0]] = np.nan
            # vals[vals<0] = np.nan
            vals_gs = T.monthly_vals_to_annual_val(vals,gs,method='array')
            vals_gs_dict = T.dict_zip(year_list,vals_gs)
            vals_drought_year = vals_gs_dict[drought_year]
            vals_drought_year_mean = np.nanmean(vals_drought_year)
            vals_drought_year_mean_list.append(vals_drought_year_mean)
        df[f'{data_name}'] = vals_drought_year_mean_list
        return df

    def add_Soil_SILT(self,df):
        # df = Load_dataframe()
        fpath = join(data_root,'HWSD/tif_05/SILT.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'SILT')
        return df

    def add_Soil_sand(self,df):
        # df = Load_dataframe()
        fpath = join(data_root,'SOIL_Grid/sand/sand.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'sand')
        return df

    def add_detrend_T_anomaly(self,df):
        # df = Load_dataframe()
        vals_spatial_dict,data_name,valid_range = Load_Data().Temperature_anomaly_detrend()
        # print(data_name)
        # exit()
        year_list = global_year_range_list
        gs = global_gs
        vals_drought_year_mean_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            # print(drought_year)
            # exit()
            vals = vals_spatial_dict[pix]
            vals = np.array(vals,dtype=float)
            vals[vals>valid_range[1]] = np.nan
            vals[vals<valid_range[0]] = np.nan
            # vals[vals<0] = np.nan
            vals_gs = T.monthly_vals_to_annual_val(vals,gs,method='array')
            vals_gs_dict = T.dict_zip(year_list,vals_gs)
            vals_drought_year = vals_gs_dict[drought_year]
            vals_drought_year_mean = np.nanmean(vals_drought_year)
            vals_drought_year_mean_list.append(vals_drought_year_mean)
        df[f'{data_name}'] = vals_drought_year_mean_list
        return df

    def add_delta_Topt_T(self,df):
        delta = df['max_Temperature-origin'] - df['optimal_temp']
        df['delta_Topt_T'] = delta
        return df
        pass

    def add_NDVI(self, df):
        # df = Load_dataframe()
        vals_spatial_dict, data_name, valid_range = Load_Data().NDVI_anomaly_detrend()
        # print(data_name)
        # exit()
        year_list = global_year_range_list
        gs = global_gs
        vals_drought_year_mean_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            # print(drought_year)
            # exit()
            vals = vals_spatial_dict[pix]
            vals = np.array(vals, dtype=float)
            vals[vals > valid_range[1]] = np.nan
            vals[vals < valid_range[0]] = np.nan
            # vals[vals<0] = np.nan
            vals_gs = T.monthly_vals_to_annual_val(vals, gs, method='array')
            vals_gs_dict = T.dict_zip(year_list, vals_gs)
            vals_drought_year = vals_gs_dict[drought_year]
            vals_drought_year_mean = np.nanmean(vals_drought_year)
            vals_drought_year_mean_list.append(vals_drought_year_mean)
        df[f'{data_name}'] = vals_drought_year_mean_list
        return df

    def add_NDVI_with_trend(self, df):
        # df = Load_dataframe()
        vals_spatial_dict, data_name, valid_range = Load_Data().NDVI_anomaly_with_trend()
        # print(data_name)
        # exit()
        year_list = global_year_range_list
        gs = global_gs
        vals_drought_year_mean_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            # print(drought_year)
            # exit()
            vals = vals_spatial_dict[pix]
            vals = np.array(vals, dtype=float)
            vals[vals > valid_range[1]] = np.nan
            vals[vals < valid_range[0]] = np.nan
            # vals[vals<0] = np.nan
            vals_gs = T.monthly_vals_to_annual_val(vals, gs, method='array')
            vals_gs_dict = T.dict_zip(year_list, vals_gs)
            vals_drought_year = vals_gs_dict[drought_year]
            vals_drought_year_mean = np.nanmean(vals_drought_year)
            vals_drought_year_mean_list.append(vals_drought_year_mean)
        df[f'{data_name}'] = vals_drought_year_mean_list
        return df

    def check_variable(self,df):
        T.print_head_n(df, 10)
        df_pix_dict = T.df_groupby(df, 'pix')
        spatial_dict = {}
        for pix in df_pix_dict:
            df_pix = df_pix_dict[pix]
            delta_Topt_T = df_pix['delta_Topt_T'].tolist()[0]
            spatial_dict[pix] = delta_Topt_T
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        plt.imshow(arr,interpolation='nearest',vmin=-5,vmax=5,cmap='RdBu_r')
        plt.colorbar()
        plt.show()
        pass

    def check_Topt_vs_ndvi(self,df):
        df = self.clean_df(df)
        ELI_class_list = global_ELI_class_list
        for ELI in ELI_class_list:
            df_ELI = df[df['ELI_class'] == ELI]
            delta_Topt_T = df_ELI['delta_Topt_T'].tolist()
            NDVI_anomaly_detrend = df_ELI['NDVI-anomaly_detrend'].tolist()
            NDVI_anomaly_detrend = df_ELI['NDVI-anomaly_with_trend'].tolist()
            KDE_plot().plot_scatter(delta_Topt_T,NDVI_anomaly_detrend)
            plt.title(f'{ELI}')
            plt.show()

        pass

    def plot_variables(self,df):
        df = df[df['drought_type']=='hot-drought']
        # df = df[df['drought_type']=='normal-drought']
        df_pix_dict = T.df_groupby(df, 'pix')
        # col = 'delta_Topt_T'
        # col = 'NDVI-anomaly_with_trend'
        # col = 'NDVI-anomaly_detrend'
        # col = 'Temperature-anomaly_detrend'
        # col = 'VPD-anomaly'
        col = 'mean_Temperature-origin'
        spatial_dict = {}
        for pix in tqdm(df_pix_dict):
            df_pix = df_pix_dict[pix]
            vals = df_pix[col].tolist()
            vals_mean = np.nanmean(vals)
            spatial_dict[pix] = vals_mean
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        arr_flatten = arr.flatten()
        plt.hist(arr_flatten,bins=100)
        plt.figure()
        plt.imshow(arr,interpolation='nearest',cmap='RdBu_r') # temperature origin
        # plt.imshow(arr,interpolation='nearest',cmap='RdBu_r',vmin=-1,vmax=1) # temperature vpd
        # plt.imshow(arr,interpolation='nearest',cmap='RdBu',vmin=-1,vmax=1) # ndvi
        # plt.imshow(arr,interpolation='nearest',cmap='RdBu_r',vmin=-5,vmax=5) # delta_Topt_T
        plt.colorbar()
        plt.title(f'{col}')
        plt.show()

        pass

    def print_drought_events_numbers(self,df):

        total_number = len(df)
        total_number_str = 'total_number'
        print(f'{total_number_str:22s}{str(total_number):6s}')
        df_hot_drought = df[df['drought_type'] == 'hot-drought']
        hot_drought_number = len(df_hot_drought)
        hot_drought_number_str = 'hot_drought_number'
        print(f'{hot_drought_number_str:22s}{str(hot_drought_number):6s}')

        df_normal_drought = df[df['drought_type'] == 'normal-drought']
        normal_drought_number = len(df_normal_drought)
        normal_drought_number_str = 'normal_drought_number'
        print(f'{normal_drought_number_str:22s}{str(normal_drought_number):6s}')

        df_dryland = df[df['AI_class'] == 'Arid']
        dryland_number = len(df_dryland)
        dryland_number_str = 'dryland_number'
        print(f'{dryland_number_str:22s}{str(dryland_number):6s}')

        df_dryland_hot_drought = df_dryland[df_dryland['drought_type'] == 'hot-drought']
        hot_drought_number = len(df_dryland_hot_drought)
        hot_drought_number_str = 'hot_drought_number'
        print(f'{hot_drought_number_str:22s}{str(hot_drought_number):6s}')

        df_dryland_normal_drought = df_dryland[df_dryland['drought_type'] == 'normal-drought']
        normal_drought_number = len(df_dryland_normal_drought)
        normal_drought_number_str = 'normal_drought_number'
        print(f'{normal_drought_number_str:22s}{str(normal_drought_number):6s}')

        pass

class Random_forests:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Random_forests', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe', 'dataframe.df')
        pass

    def run(self):
        # self.copy_df()
        df = self.__gen_df_init()

        # T.print_head_n(df)


        # self.run_importance(df)
        self.plot_importance()

        # self.run_partial_dependence_plots(df)
        # self.plot_PDP()

        pass

    def copy_df(self):
        T.mkdir(join(self.this_class_arr, 'dataframe'))
        if isfile(self.dff):
            print('already exists: ', self.dff)
            print('press enter to overwrite')
            pause()
            pause()
            pause()
        T.mkdir(join(self.this_class_arr, 'dataframe'))
        dff = join(Attribution_Dataframe().this_class_arr,  'dataframe', 'dataframe.df')
        df = T.load_df(dff)
        T.save_df(df,self.dff)
        T.df_to_excel(df, self.dff)

    def __gen_df_init(self):
        if not os.path.isfile(self.dff):
            df = pd.DataFrame()
            T.save_df(df,self.dff)
            return df
        else:
            df,dff = self.__load_df()
            return df

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print('len(df):',len(df))
        return df,dff


    def run_importance(self,df):
        outdir = join(self.this_class_arr,'importance')
        T.mk_dir(outdir)
        x_variables,y_variable = Attribution_Dataframe().variables_info()
        df = Attribution_Dataframe().clean_df(df)
        df = df.dropna(subset=[y_variable])
        df = df.dropna(subset=x_variables)
        drought_type_list = global_drought_type_list
        ELI_list = global_ELI_class_list
        for ELI in ELI_list:
            df_ELI = df[df['ELI_class'] == ELI]
            # for drt in drought_type_list:
            # df_drt = df_ELI[df_ELI['drought_type'] == drt]

            X = df_ELI[x_variables]
            Y = df_ELI[y_variable]
            clf, importances_dic, mse, r_model, score, Y_test, y_pred = (
                self._random_forest_train(X, Y, x_variables))
            result_dic = {
                'importances_dic':importances_dic,
                'r_model':r_model,
                'score':score,
            }
            # outf = join(outdir, f'{drt}_{ELI}.npy')
            outf = join(outdir, f'{ELI}.npy')
            T.save_npy(result_dic, outf)


    def _random_forest_train(self, X, Y, variable_list):
        '''
        :param X: a dataframe of x variables
        :param Y: a dataframe of y variable
        :param variable_list: a list of x variables
        :return: details of the random forest model and the importance of each variable
        '''
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) # split the data into training and testing
        clf = RandomForestRegressor(n_estimators=100, n_jobs=24) # build a random forest model
        clf.fit(X_train, Y_train) # train the model
        result = permutation_importance(clf, X_train, Y_train, scoring=None,
                                        n_repeats=50, random_state=1,
                                        n_jobs=1) # calculate the importance of each variable using permutation importance
        importances = result.importances_mean # get the importance of each variable
        importances_dic = dict(zip(variable_list, importances)) # put the importance of each variable into a dictionary
        labels = []
        importance = []
        for key in variable_list:
            labels.append(key)
            importance.append(importances_dic[key])
        y_pred = clf.predict(X_test) # predict the y variable using the testing data
        r_model = stats.pearsonr(Y_test, y_pred)[0] # calculate the correlation between the predicted y variable and the actual y variable
        mse = sklearn.metrics.mean_squared_error(Y_test, y_pred) # calculate the mean squared error
        score = clf.score(X_test, Y_test) # calculate the R^2
        return clf, importances_dic, mse, r_model, score, Y_test, y_pred

    def plot_importance(self):
        fdir = join(self.this_class_arr, 'importance')
        outdir = join(self.this_class_png, 'importance')
        T.mk_dir(outdir, force=True)
        ELI_class_list = global_ELI_class_list
        # season_list = global_drought_season_list
        x_list = []
        y_list = []
        title_list = []
        for ELI in ELI_class_list:
            f = f'{ELI}.npy'
            fpath = join(fdir, f)
            result_dict = T.load_npy(fpath)
            importances_dic = result_dict['importances_dic']
            r_model = result_dict['r_model']
            score = result_dict['score']
            title = f'{f}\nR^2={score:.2f}, r={r_model:.2f}'
            x = importances_dic.keys()
            x = list(x)
            y = [importances_dic[key] for key in x]
            y_sort_index = np.argsort(y)
            y_sort = [y[i] for i in y_sort_index]
            x_sort = [x[i] for i in y_sort_index]
            # print(x_sort)
            # exit()
            x_list.append(x_sort)
            y_list.append(y_sort)
            title_list.append(title)
        y_list = np.array(y_list)
        # print(y_list)
        # print(x_list)
        # exit()
        # y_list_mean = np.nanmean(y_list,axis=0)
        # sort
        # y_list_mean_sort = np.argsort(y_list_mean)
        # x_list_sort = [x_list[0][i] for i in y_list_mean_sort]
        # flag = 0
        for i in range(len(x_list)):
            # plt.figure(figsize=(5 * centimeter_factor, 10 * centimeter_factor))
            plt.figure()
            print(x_list[i])
            print(y_list[i])
            plt.scatter(y_list[i],x_list[i],s=100,marker='o',c='k',linewidths=2,)
            plt.title(title_list[i])
            # flag += 1
            # plt.xlim(0, 1)


            # plt.title(f'{drt}')
            # plt.legend()
            outf = join(outdir, f'{ELI_class_list[i]}.pdf')
            plt.tight_layout()

            plt.savefig(outf, dpi=300)
            plt.close()
        # plt.show()

    def run_partial_dependence_plots(self,df):
        # fdir = join(Random_Forests_delta().this_class_arr, 'seasonal_delta')
        outdir = join(self.this_class_arr, 'partial_dependence_plots')
        T.mk_dir(outdir, force=True)
        x_variables, y_variable = Attribution_Dataframe().variables_info()
        df = Attribution_Dataframe().clean_df(df)
        ELI_class_list = global_ELI_class_list
        drought_type_list = global_drought_type_list

        # for drt in drought_type_list:
        #     df_drt = df[df['drought_type'] == drt]
        for ELI in ELI_class_list:
            df_ELI = df[df['ELI_class']==ELI]

            result_dic = self.partial_dependence_plots(df_ELI, x_variables, y_variable)
            # outf = join(outdir, f'{drt}_{ELI}.npy')
            outf = join(outdir, f'{ELI}.npy')
            T.save_npy(result_dic, outf)

    def plot_PDP(self):
        fdir = join(self.this_class_arr,'partial_dependence_plots')
        outdir = join(self.this_class_png,'partial_dependence_plots')
        T.mk_dir(outdir,force=True)
        ELI_class_list = global_ELI_class_list
        drought_type_list = global_drought_type_list
        for ELI in ELI_class_list:
            # plt.figure(figsize=(7, 7))
            # for drt in drought_type_list:
            fpath = join(fdir,f'{ELI}.npy')

            result_dict = T.load_npy(fpath)
            flag = 1
            for key in result_dict:
                result_dict_i = result_dict[key]
                x = result_dict_i['x']
                y = result_dict_i['y']
                y_std = result_dict_i['y_std']
                plt.subplot(3,3,flag)
                flag += 1
                y = SMOOTH().smooth_convolve(y,window_len=5)
                plt.plot(x,y,label=ELI)
                # y_std = y_std / 4
                # plt.fill_between(x,y-y_std,y+y_std,alpha=0.5)
                # plt.legend()
                # plt.ylim(-.5,.5)
                plt.xlabel(key.replace('_vs_NDVI-anomaly_detrend_','\nsensitivity\n'))
            plt.legend()
            # plt.suptitle(ELI)

            plt.tight_layout()
            # outf = join(outdir,f'{season}.pdf')
            # plt.savefig(outf,dpi=300)
        plt.show()
        # T.open_path_and_file(outdir)

    def partial_dependence_plots(self,df,x_vars,y_var):
        '''
        :param df: a dataframe
        :param x_vars: a list of x variables
        :param y_var: a y variable
        :return:
        '''
        all_vars = copy.copy(x_vars) # copy the x variables
        all_vars.append(y_var) # add the y variable to the list
        all_vars_df = df[all_vars] # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna() # drop rows with missing values
        X = all_vars_df[x_vars]
        Y = all_vars_df[y_var]
        model, r2 = self.__train_model(X, Y) # train a Random Forests model
        flag = 0
        result_dic = {}
        for var in x_vars:
            flag += 1
            df_PDP = self.__get_PDPvalues(var, X, model) # get the partial dependence plot values
            ppx = df_PDP[var]
            ppy = df_PDP['PDs']
            ppy_std = df_PDP['PDs_std']
            result_dic[var] = {'x':ppx,
                               'y':ppy,
                               'y_std':ppy_std,
                               'r2':r2}
        return result_dic

    def __train_model(self,X,y):
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, random_state=1, test_size=0.0) # split the data into training and testing
        rf = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=20) # build a random forest model
        rf.fit(X, y) # train the model
        # r2 = rf.score(X_test,y_test)
        return rf,0.999

    def __get_PDPvalues(self, col_name, data, model, grid_resolution=50):
        '''
        :param col_name: a variable
        :param data: a dataframe of x variables
        :param model: a random forest model
        :param grid_resolution: the number of points in the partial dependence plot
        :return: a dataframe of the partial dependence plot values
        '''
        Xnew = data.copy()
        sequence = np.linspace(np.min(data[col_name]), np.max(data[col_name]), grid_resolution) # create a sequence of values
        Y_pdp = []
        Y_pdp_std = []
        for each in sequence:
            Xnew[col_name] = each
            Y_temp = model.predict(Xnew)
            Y_pdp.append(np.mean(Y_temp))
            Y_pdp_std.append(np.std(Y_temp))
        return pd.DataFrame({col_name: sequence, 'PDs': Y_pdp, 'PDs_std': Y_pdp_std})

class SHAP:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('SHAP_with_soil', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe', 'dataframe.df')
        pass

    def run(self):
        # self.copy_df()
        df = self.__gen_df_init()
        # self.pdp_shap(df)
        self.plot_pdp_shap_result_line()
        # self.plot_pdp_shap_result_scatter(df)
        # self.pdp_shap_split_df(df)
        # self.plot_pdp_shap_split_df_scatter()
        # self.plot_pdp_shap_split_df_line()
        # self.plot_pdp_shap_split_df_line_breakpoints()
        # self.plot_pdp_shap_split_df_line_breakpoints_all_regions()
        # self.plot_pdp_shap_split_df_line_breakpoints_detail()
        # self.plot_pdp_shap_split_df_drought_mon()
        # self.plot_importances()
        pass

    def copy_df(self):
        T.mkdir(join(self.this_class_arr, 'dataframe'))
        if isfile(self.dff):
            print('already exists: ', self.dff)
            print('press enter to overwrite')
            pause()
            pause()
            pause()
        T.mkdir(join(self.this_class_arr, 'dataframe'))
        dff = join(Attribution_Dataframe().this_class_arr,  'dataframe', 'dataframe.df')
        df = T.load_df(dff)
        T.save_df(df,self.dff)
        T.df_to_excel(df, self.dff)

    def __gen_df_init(self):
        T.mkdir(join(self.this_class_arr, 'dataframe'))
        if not os.path.isfile(self.dff):
            df = pd.DataFrame()
            T.save_df(df,self.dff)
            return df
        else:
            df,dff = self.__load_df()
            return df

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print('len(df):',len(df))
        return df,dff

    def __ylim(self):
        # ('Energy-Limited', 'Water-Limited')
        lim_dict = {
            'Water-Limited':{
                'SOS':(-.05,.05),
                'VPD-anomaly':(-.1,.1),
                'Temperature-anomaly_detrend':(-.1,.1),
                'Temperature-anomaly':(-.1,.1),
                'Temperature_quantile':(-.1,.1),
                'FAPAR-anomaly_detrend':(-.2,.1),
                'Radiation-anomaly':(-.2,.1),
                'drought_mon':(-.1,.1),
                'GLEAM-SMRoot-anomaly_detrend':(-.2,.3),
                'SILT':(-.05,.05),
                'sand':(-.05,.05),
            },
            'Energy-Limited':{
                'SOS':(-.1,.1),
                'VPD-anomaly':(-.25,.1),
                'Temperature-anomaly_detrend':(-.05,.1),
                'Temperature-anomaly':(-.05,.1),
                'Temperature_quantile':(-.05,.1),
                'FAPAR-anomaly_detrend':(-.2,.1),
                'Radiation-anomaly':(-.1,.1),
                'drought_mon':(-.1,.1),
                'GLEAM-SMRoot-anomaly_detrend':(-.1,.1),
                'SILT':(-.05,.05),
                'sand':(-.1,.1),
            }
        }
        return lim_dict


    def pdp_shap(self,df):

        x_variables,y_variable = Attribution_Dataframe().variables_info()
        df = Attribution_Dataframe().clean_df(df)
        ELI_class_list = global_ELI_class_list
        for ELI in ELI_class_list:
            outdir = join(self.this_class_arr, 'pdp_shap', str(ELI))
            # outf = join(outdir,self.y_variable)
            T.mk_dir(outdir, force=True)
            df_ELI = df[df['ELI_class']==ELI]

            X = df_ELI[x_variables]
            Y = df_ELI[y_variable]
            model,y,y_pred = self.__train_model(X, Y)  # train a Random Forests model
            imp_dict_xgboost = {}
            for i in range(len(x_variables)):
                imp_dict_xgboost[x_variables[i]] = model.feature_importances_[i]
            sorted_imp = sorted(imp_dict_xgboost.items(), key=lambda x: x[1], reverse=True)
            imp_dict_outf = join(outdir, 'imp_xgboost')
            T.save_npy(imp_dict_xgboost, imp_dict_outf)
            x_ = []
            y_ = []
            for key, value in sorted_imp:
                x_.append(key)
                y_.append(value)
            explainer = shap.TreeExplainer(model)
            y_base = explainer.expected_value
            print('y_base', y_base)
            print('y_mean', np.mean(y))
            shap_values = explainer(X)
            imp_dict = self.__feature_importances_shap_values(shap_values, x_variables)
            outf_impdict_shap = join(outdir, 'shaply_imp_dict')
            T.save_npy(imp_dict, outf_impdict_shap)

            outf_shap_values = join(outdir, 'shaply_shap_values')
            T.save_dict_to_binary(shap_values, outf_shap_values)
            # T.save_npy(shap_values, outf_shap_values)

    def plot_pdp_shap_result_line(self):
        ELI_class_list = global_ELI_class_list
        # ('Energy-Limited', 'Water-Limited')
        # ylim_dict = {
        #     'Energy-Limited': [-0.1, 0.1],
        #     'Water-Limited': [-0.3, 0.3],
        # }
        for ELI in ELI_class_list:

            fdir = join(self.this_class_arr, 'pdp_shap', str(ELI))
            outdir = join(self.this_class_png, 'pdp_shap', str(ELI))
            T.mk_dir(outdir, force=True)
            imp_dict_fpath = join(fdir, 'shaply_imp_dict.npy')
            shap_values_fpath = join(fdir, 'shaply_shap_values.pkl')
            shap_values = T.load_dict_from_binary(shap_values_fpath)
            # exit()
            imp_dict = T.load_npy(imp_dict_fpath)
            x_list = []
            y_list = []
            for key in imp_dict.keys():
                x_list.append(key)
                y_list.append(imp_dict[key])


            flag = 1
            plt.figure(figsize=(18 * centimeter_factor, 9 * centimeter_factor))
            for x_var in x_list:
                print(x_var)
                shap_values_mat = shap_values[:, x_var]
                outf_i = join(outdir, f'shaply_{x_var}')
                # T.save_npy(shap_values_mat, outf_i)
                data_i = shap_values_mat.data
                value_i = shap_values_mat.values
                df_i = pd.DataFrame({x_var: data_i, 'shap_v': value_i})
                # df_i_random = df_i.sample(n=len(df_i) // 2)
                # df_i = df_i_random

                # x_variable_range_dict = self.x_variable_range_dict
                # start,end = x_variable_range_dict[x_var]
                if not x_var == 'drought_mon':
                    start, end = Attribution_Dataframe().variables_threshold()[x_var]
                    bins = np.linspace(start, end, 50)
                    df_group, bins_list_str = T.df_bin(df_i, x_var, bins)
                    y_mean_list = []
                    x_mean_list = []
                    y_err_list = []
                    scatter_x_list = df_i[x_var].tolist()
                    scatter_y_list = df_i['shap_v'].tolist()
                    x_q10 = np.quantile(scatter_x_list, 0.1)
                    x_q90 = np.quantile(scatter_x_list, 0.9)
                    for name, df_group_i in df_group:
                        x_i = name[0].left
                        # print(x_i)
                        # exit()
                        vals = df_group_i['shap_v'].tolist()

                        if len(vals) == 0:
                            continue
                        # mean = np.nanmean(vals)
                        mean = np.nanmedian(vals)
                        err = np.nanstd(vals)
                        y_mean_list.append(mean)
                        x_mean_list.append(x_i)
                        y_err_list.append(err)
                else:
                    x_unique = df_i[x_var].unique()

                    x_unique = list(x_unique)
                    x_unique.sort()
                    y_vals_list = []
                    for x_i in x_unique:
                        y_i = df_i[df_i[x_var] == x_i]['shap_v'].tolist()
                        y_vals_list.append(y_i)
                plt.subplot(3, 3, flag)
                # print(data_i[0])
                # exit()
                # interp_model = interpolate.interp1d(x_mean_list, y_mean_list, kind='cubic')
                # y_interp = interp_model(x_mean_list)
                if x_var == 'drought_mon':
                    plt.boxplot(y_vals_list, positions=x_unique, showfliers=False, showmeans=False)
                    pass
                else:
                    y_mean_list = SMOOTH().smooth_convolve(y_mean_list, window_len=7)
                    plt.plot(x_mean_list, y_mean_list, c='red', alpha=1)
                    var_min = self.__variables_ranges()[ELI][x_var][0]
                    var_max = self.__variables_ranges()[ELI][x_var][1]
                    plt.xlim(var_min, var_max)
                    up, bottom = self.__ylim()[ELI][x_var]
                    plt.vlines(x_q10, ymin=bottom, ymax=up, colors='gray', linestyles='dashed', zorder=-1)
                    plt.vlines(x_q90, ymin=bottom, ymax=up, colors='gray', linestyles='dashed', zorder=-1)
                    # plt.scatter(scatter_x_list, scatter_y_list, alpha=0.2, c='gray', marker='.', s=1, zorder=-1)

                plt.xlabel(x_var)
                flag += 1
                # plt.ylim(-0.1, 0.1)
                up,bottom = self.__ylim()[ELI][x_var]
                plt.ylim(up,bottom)


            # plt.suptitle(y_variable)
            plt.tight_layout()
            # plt.show()
            outf = join(outdir, 'shaply.pdf')
            # outf = join(outdir, 'shaply.png')
            # plt.show()
            plt.savefig(outf, dpi=300)
            plt.close()
        T.open_path_and_file(outdir)


    def plot_pdp_shap_result_scatter(self,df):
        df_clean = Attribution_Dataframe().clean_df(df)
        ELI_class_list = global_ELI_class_list
        variables_range_dict = Attribution_Dataframe().variables_threshold()
        ylim_dict = self.__ylim()
        drought_type_zorder_dict = {
            'hot-drought':99,
            'normal-drought':0
        }
        drought_type_color_dict = {
            'normal-drought': 'b',
            'hot-drought': 'r',
        }
        drought_type_alpha_dict = {
            'normal-drought': 1,
            'hot-drought': .2,
        }
        drought_type_size_dict = {
            'normal-drought': 1,
            'hot-drought': .5,
        }

        for ELI in ELI_class_list:
            df_ELI = df_clean[df_clean['ELI_class'] == ELI]
            # T.print_head_n(df_ELI)
            drought_type_list = df_ELI['drought_type'].tolist()
            fdir = join(self.this_class_arr, 'pdp_shap', str(ELI))
            outdir = join(self.this_class_png, 'pdp_shap', str(ELI))
            # outdir = join(self.this_class_png, 'pdp_shap_split', str(ELI))
            T.mk_dir(outdir, force=True)
            imp_dict_fpath = join(fdir, 'shaply_imp_dict.npy')
            shap_values_fpath = join(fdir, 'shaply_shap_values.pkl')
            shap_values = T.load_dict_from_binary(shap_values_fpath)
            # exit()
            imp_dict = T.load_npy(imp_dict_fpath)
            x_list = []
            y_list = []
            for key in imp_dict.keys():
                x_list.append(key)
                y_list.append(imp_dict[key])

            flag = 1
            for x_var in x_list:
                print(ELI, x_var)

                shap_values_mat = shap_values[:, x_var]
                outf_i = join(outdir, f'shaply_{x_var}')
                # T.save_npy(shap_values_mat, outf_i)
                data_i = shap_values_mat.data
                # print('data_i',len(data_i))
                # exit()
                value_i = shap_values_mat.values
                df_i = pd.DataFrame({x_var: data_i, 'shap_v': value_i, 'drought_type': drought_type_list})

                if x_var == 'drought_mon':
                    plt.figure(figsize=(9 * centimeter_factor, 6 * centimeter_factor))

                    df_mon_group = T.df_groupby(df_i,x_var)
                    x_list = []
                    y_list = []
                    for x in df_mon_group:
                        df_mon_group_i = df_mon_group[x]
                        for drt in global_drought_type_list:
                            df_mon_group_i_drt = df_mon_group_i[df_mon_group_i['drought_type'] == drt]
                            scatter_y_list = df_mon_group_i_drt['shap_v'].tolist()
                            x_list.append(f'{drt}{x}')
                            y_list.append(scatter_y_list)
                    plt.boxplot(y_list, labels=x_list, showfliers=False, showmeans=False)
                    plt.xlabel(x_var)
                    flag += 1
                    up,bottom = ylim_dict[ELI][x_var]
                    plt.ylim(up,bottom)
                    plt.title(f'{x_var}')
                    outf = join(outdir, f'{x_var}.pdf')
                    plt.show()
                    # plt.savefig(outf, dpi=900)
                    # plt.close()
                else:
                    # continue
                    plt.figure(figsize=(9 * centimeter_factor, 6 * centimeter_factor))

                    for drt in global_drought_type_list:
                        df_i_drt = df_i[df_i['drought_type'] == drt]
                        scatter_x_list = df_i_drt[x_var].tolist()

                        # plt.hist(scatter_x_list)
                        # plt.show()
                        scatter_y_list = df_i_drt['shap_v'].tolist()
                        plt.scatter(scatter_x_list, scatter_y_list, alpha=drought_type_alpha_dict[drt], c=drought_type_color_dict[drt],
                            zorder=drought_type_zorder_dict[drt], marker='.', s=drought_type_size_dict[drt])
                        var_min = self.__variables_ranges()[ELI][x_var][0]
                        var_max = self.__variables_ranges()[ELI][x_var][1]
                        plt.xlim(var_min, var_max)
                    plt.xlabel(x_var)
                    flag += 1
                    up, bottom = ylim_dict[ELI][x_var]
                    plt.ylim(up, bottom)
                    plt.title(f'{x_var}')
                    outf = join(outdir, f'{x_var}.png')
                    plt.show()
                    # plt.savefig(outf, dpi=900)
                    # plt.close()
            pass

    def pdp_shap_split_df(self,df):
        outdir = join(self.this_class_arr, 'pdp_shap_split1')
        df_clean = Attribution_Dataframe().clean_df(df)
        ELI_class_list = global_ELI_class_list
        drought_type_color_dict = global_drought_type_color_dict
        for ELI in ELI_class_list:
            df_ELI = df_clean[df_clean['ELI_class'] == ELI]
            T.print_head_n(df_ELI)
            drought_type_list = df_ELI['drought_type'].tolist()

            fdir = join(self.this_class_arr, 'pdp_shap', str(ELI))
            outdir_i = join(outdir, str(ELI))
            # outdir = join(self.this_class_png, 'pdp_shap_split', str(ELI))
            T.mk_dir(outdir_i, force=True)
            imp_dict_fpath = join(fdir, 'shaply_imp_dict.npy')
            shap_values_fpath = join(fdir, 'shaply_shap_values.pkl')
            shap_values = T.load_dict_from_binary(shap_values_fpath)
            # exit()
            imp_dict = T.load_npy(imp_dict_fpath)
            x_list = []
            y_list = []
            for key in imp_dict.keys():
                x_list.append(key)
                y_list.append(imp_dict[key])

            for x_var in tqdm(x_list,desc=f'{ELI}'):

                shap_values_mat = shap_values[:, x_var]
                outf_i = join(outdir_i, f'shaply_{x_var}.df')
                data_i = shap_values_mat.data
                value_i = shap_values_mat.values
                df_i = pd.DataFrame({x_var: data_i, 'shap_v': value_i, 'drought_type': drought_type_list})
                T.save_df(df_i, outf_i)
                T.df_to_excel(df_i, outf_i)
            pass

    def plot_pdp_shap_split_df_scatter(self):
        fdir = join(self.this_class_arr, 'pdp_shap_split1')
        drt_list = global_drought_type_list

        for ELI in global_ELI_class_list:
            fdir_i = join(fdir, str(ELI))
            outdir_i = join(self.this_class_png, 'pdp_shap_split_df_scatter1', str(ELI))
            T.mk_dir(outdir_i, force=True)
            for f in T.listdir(fdir_i):
                if not f.endswith('.df'):
                    continue
                var_name = f.split('.')[0].replace('shaply_', '')
                fpath = join(fdir_i, f)
                df = T.load_df(fpath)
                # T.print_head_n(df)
                start, end = Attribution_Dataframe().variables_threshold()[var_name]
                bins = np.linspace(start, end, 50)

                for drt in drt_list:
                    df_i = df[df['drought_type'] == drt]
                    df_group, bins_list_str = T.df_bin(df_i, var_name, bins)
                    # T.print_head_n(df)
                    x_mean_list = []
                    y_mean_list = []
                    y_err_list = []
                    for name, df_group_i in df_group:
                        x_i = name[0].left
                        # print(x_i)
                        # exit()
                        vals = df_group_i['shap_v'].tolist()

                        if len(vals) == 0:
                            continue
                        # mean = np.nanmean(vals)
                        mean = np.nanmedian(vals)
                        err = np.nanstd(vals)
                        y_mean_list.append(mean)
                        x_mean_list.append(x_i)
                        y_err_list.append(err)
                    # plt.plot(x_mean_list, y_mean_list, label=drt,zorder=3)
                    x_vals = df_i[var_name].tolist()
                    y_vals = df_i['shap_v'].tolist()
                    color = global_drought_type_color_dict[drt]
                    if drt == 'hot-drought':
                        zorder = 2
                    else:
                        zorder = 1
                    plt.scatter(x_vals, y_vals, color=color, alpha=0.1, zorder=zorder,linewidths=0)
                    # plt.scatter(x_vals, y_vals, color=color, alpha=0.5,linewidths=0)
                # plt.legend()
                plt.title(f'{ELI}\n{var_name}')
                # plt.xlim(start, end)
                plt.ylim(-0.6, 0.6)
                # plt.show()

                plt.savefig(join(outdir_i, f'{var_name}.png'))
                plt.close()

    def plot_pdp_shap_split_df_line(self):
        fdir = join(self.this_class_arr, 'pdp_shap_split1')
        drt_list = global_drought_type_list

        for ELI in global_ELI_class_list:
            fdir_i = join(fdir, str(ELI))
            outdir_i = join(self.this_class_png, 'pdp_shap_split_df_scatter1', str(ELI))
            T.mk_dir(outdir_i, force=True)
            for f in T.listdir(fdir_i):
                if not f.endswith('.df'):
                    continue
                var_name = f.split('.')[0].replace('shaply_', '')
                fpath = join(fdir_i, f)
                df = T.load_df(fpath)
                # T.print_head_n(df)
                start, end = Attribution_Dataframe().variables_threshold()[var_name]
                bins = np.linspace(start, end, 50)

                for drt in drt_list:
                    df_i = df[df['drought_type'] == drt]
                    df_group, bins_list_str = T.df_bin(df_i, var_name, bins)
                    # T.print_head_n(df)
                    x_mean_list = []
                    y_mean_list = []
                    y_err_list = []
                    for name, df_group_i in df_group:
                        x_i = name[0].left
                        # print(x_i)
                        # exit()
                        vals = df_group_i['shap_v'].tolist()

                        if len(vals) == 0:
                            continue
                        # mean = np.nanmean(vals)
                        mean = np.nanmedian(vals)
                        err = np.nanstd(vals)
                        y_mean_list.append(mean)
                        x_mean_list.append(x_i)
                        y_err_list.append(err)
                    plt.plot(x_mean_list, y_mean_list, label=drt,zorder=3)
                    x_vals = df_i[var_name].tolist()
                    y_vals = df_i['shap_v'].tolist()
                    color = global_drought_type_color_dict[drt]
                    if drt == 'hot-drought':
                        zorder = 2
                    else:
                        zorder = 1
                    # plt.scatter(x_vals, y_vals, color=color, alpha=0.1, zorder=zorder,linewidths=0)
                    # plt.scatter(x_vals, y_vals, color=color, alpha=0.5,linewidths=0)
                # plt.legend()
                plt.title(f'{ELI}\n{var_name}')
                # plt.xlim(start, end)
                plt.ylim(-0.6, 0.6)
                plt.show()

                # plt.savefig(join(outdir_i, f'{var_name}.pdf'))
                # plt.close()

    def plot_pdp_shap_split_df_line_breakpoints(self):

        fdir = join(self.this_class_arr, 'pdp_shap_split')
        drt_list = global_drought_type_list

        for ELI in global_ELI_class_list:
            fdir_i = join(fdir, str(ELI))
            outdir_i = join(self.this_class_png, 'plot_pdp_shap_split_df_line_breakpoints', str(ELI))
            T.mk_dir(outdir_i, force=True)
            for f in T.listdir(fdir_i):
                if not f.endswith('.df'):
                    continue
                var_name = f.split('.')[0].replace('shaply_', '')
                fpath = join(fdir_i, f)
                df = T.load_df(fpath)
                # T.print_head_n(df)
                start, end = Attribution_Dataframe().variables_threshold()[var_name]
                bins = np.linspace(start, end, 100)

                for drt in drt_list:
                    df_i = df[df['drought_type'] == drt]
                    df_group, bins_list_str = T.df_bin(df_i, var_name, bins)
                    # T.print_head_n(df)
                    x_mean_list = []
                    y_mean_list = []
                    y_err_list = []
                    for name, df_group_i in df_group:
                        x_i = name[0].left
                        # print(x_i)
                        # exit()
                        vals = df_group_i['shap_v'].tolist()

                        if len(vals) == 0:
                            continue
                        # mean = np.nanmean(vals)
                        mean = np.nanmedian(vals)
                        err = np.nanstd(vals)
                        y_mean_list.append(mean)
                        x_mean_list.append(x_i)
                        y_err_list.append(err)
                    y_mean_list = np.array(y_mean_list)
                    x_mean_list = np.array(x_mean_list)
                    # try:
                    #     y_mean_list = SMOOTH().mid_window_smooth(y_mean_list, window=5)
                    # except:
                    #     continue
                    # if not 'VPD' in var_name:
                    #     continue
                    o = self.change_point_rbeast(y_mean_list,
                                                season='none',
                                                tcp_minmax=(0, 1),
                                                    )
                    # rb.plot(o, title=var_name, fig=plt.figure(figsize=(15, 10)))
                    change_point = o.trend.cp[0]
                    change_point = int(change_point)
                    # exit()
                    x_vals = df_i[var_name].tolist()
                    y_vals = df_i['shap_v'].tolist()
                    color = global_drought_type_color_dict[drt]
                    if drt == 'hot-drought':
                        zorder = 2
                    else:
                        zorder = 1
                    plt.plot(x_mean_list, y_mean_list, label=drt,zorder=3)

                    plt.scatter(x_mean_list[change_point], y_mean_list[change_point], color=color, alpha=1, zorder=100)
                    plt.vlines(x_mean_list[change_point], -0.6, 0.6, color=color, alpha=1, zorder=100)
                    # plt.scatter(x_vals, y_vals, color=color, alpha=0.1, zorder=zorder,linewidths=0)
                    # plt.scatter(x_vals, y_vals, color=color, alpha=0.5,linewidths=0)
                # plt.legend()
                print('-----')
                plt.title(f'{ELI}\n{var_name}')
                # # plt.xlim(start, end)
                plt.ylim(-0.6, 0.6)
                # plt.show()

                plt.savefig(join(outdir_i, f'{var_name}.pdf'))
                plt.close()


    def plot_pdp_shap_split_df_line_breakpoints_all_regions(self):

        fdir_energy = join(self.this_class_arr, 'pdp_shap_split1')
        fdir_water = join(self.this_class_arr, 'pdp_shap_split')
        drt_list = global_drought_type_list
        # print_fdir(fdir)
        # ('Energy-Limited', 'Water-Limited')
        for ELI in global_ELI_class_list[::-1]:
            if ELI == 'Energy-Limited':
                fdir = fdir_energy
            elif ELI == 'Water-Limited':
                fdir = fdir_water
            else:
                raise
            fdir_i = join(fdir, str(ELI))
            outdir_i = join(self.this_class_png, 'plot_pdp_shap_split_df_line_breakpoints_all_regions', str(ELI))
            T.mk_dir(outdir_i, force=True)
            for f in T.listdir(fdir_i):
                if not f.endswith('.df'):
                    continue
                var_name = f.split('.')[0].replace('shaply_', '')
                if 'drought_mon' in var_name:
                    continue
                fpath = join(fdir_i, f)
                df = T.load_df(fpath)
                # T.print_head_n(df)
                start, end = Attribution_Dataframe().variables_threshold()[var_name]
                bins = np.linspace(start, end, 100)

                df_group, bins_list_str = T.df_bin(df, var_name, bins)
                # T.print_head_n(df)
                x_mean_list = []
                y_mean_list = []
                y_err_list = []
                for name, df_group_i in df_group:
                    x_i = name[0].left
                    # print(x_i)
                    # exit()
                    vals = df_group_i['shap_v'].tolist()

                    if len(vals) < 100:
                        continue
                    # mean = np.nanmean(vals)
                    mean = np.nanmedian(vals)
                    err = np.nanstd(vals)
                    y_mean_list.append(mean)
                    x_mean_list.append(x_i)
                    y_err_list.append(err)
                y_mean_list = np.array(y_mean_list)
                x_mean_list = np.array(x_mean_list)
                o = self.change_point_rbeast(y_mean_list,
                                            season='none',
                                            tcp_minmax=(0, 1),
                                                )
                # rb.plot(o, title=var_name, fig=plt.figure(figsize=(15, 10)))
                # plt.show()
                plt.figure()
                change_point = o.trend.cp[0]
                change_point = int(change_point)
                # exit()
                x_vals = df[var_name].tolist()
                y_vals = df['shap_v'].tolist()
                plt.vlines(x_mean_list[change_point], -0.6, 0.6, alpha=1, zorder=100)
                threshold = x_mean_list[change_point]
                plt.scatter(x_vals, y_vals, alpha=0.1,linewidths=0)
                # plt.scatter(x_vals, y_vals,  alpha=0.5,linewidths=0)
                # df_i = pd.DataFrame({'x': x_mean_list, 'y': y_mean_list})
                # df_i = pd.DataFrame({'x': x_mean_list, 'y': y_mean_list})
                df_i = pd.DataFrame({'x': x_vals, 'y': y_vals})
                df_i_1 = df_i[df_i['x'] < threshold]
                df_i_2 = df_i[df_i['x'] >= threshold]
                x1_bins = np.linspace(df_i_1['x'].min(), df_i_1['x'].max(), 100)
                x2_bins = np.linspace(df_i_2['x'].min(), df_i_2['x'].max(), 100)
                df1_group, bins_list_str1 = T.df_bin(df_i_1,'x',x1_bins)
                df2_group, bins_list_str2 = T.df_bin(df_i_2,'x',x2_bins)
                err1_y_list = []
                bin1_x_list = []
                for name,df_group_i in df1_group:
                    x_vals = df_group_i['x'].tolist()
                    y_vals = df_group_i['y'].tolist()

                    mean_x = np.nanmean(x_vals)
                    mean_y = np.nanmean(y_vals)
                    bin1_x_list.append(mean_x)
                    err = np.nanstd(y_vals)
                    err1_y_list.append(err)
                err1_y_list = SMOOTH().smooth_convolve(err1_y_list,window_len=21)
                bin1_x_list = np.array(bin1_x_list)
                err2_y_list = []
                bin2_x_list = []
                for name,df_group_i in df2_group:
                    x_vals = df_group_i['x'].tolist()
                    y_vals = df_group_i['y'].tolist()

                    mean_x = np.nanmean(x_vals)
                    mean_y = np.nanmean(y_vals)
                    bin2_x_list.append(mean_x)
                    err = np.nanstd(y_vals)
                    err2_y_list.append(err)
                bin2_x_list = np.array(bin2_x_list)
                err2_y_list = SMOOTH().smooth_convolve(err2_y_list,window_len=21)
                # plt.plot(bin1_x_list,err1_y_list)
                # plt.show()
                x1 = df_i_1['x'].tolist()
                y1 = df_i_1['y'].tolist()
                x2 = df_i_2['x'].tolist()
                y2 = df_i_2['y'].tolist()
                # a, b, r, p = T.nan_line_fit(x1, y1)
                a1, b1, r1, p1 = KDE_plot().linefit(x1, y1)
                a2, b2, r2, p2 = KDE_plot().linefit(x2, y2)
                # print(x1_bins[:-1],err1_y_list)
                self.plot_fit_line(a1, b1, r1, p1, bin1_x_list,err1_y_list)
                self.plot_fit_line(a2, b2, r2, p2, bin2_x_list,err2_y_list)
                plt.title(f'{ELI}\n{var_name}')
                plt.ylim(-0.6, 0.6)

                # plt.savefig(join(outdir_i, f'{var_name}.pdf'))
                plt.savefig(join(outdir_i, f'{var_name}_with_scatter.png'),dpi=300)
                plt.close()


    def plot_fit_line(self, a, b, r, p, x, err_list,ax=None, title='', is_label=True, is_formula=True, line_color='k', **argvs):
        '''
        画拟合直线 y=ax+b
        画散点图 X,Y
        :param a:
        :param b:
        :param X:
        :param Y:
        :param i:
        :param title:
        :return:
        '''
        # x = np.linspace(min(X), max(X), len(X))
        y = a * x + b
        #
        # plt.subplot(2,2,i)
        # plt.scatter(X,Y,marker='o',s=5,c = 'grey')
        # plt.plot(X,Y)
        c = line_color
        if is_label == True:
            if is_formula == True:
                label = 'y={:0.2f}x+{:0.2f}\nr={:0.2f}\np={:0.2f}'.format(a, b, r, p)
            else:
                label = 'r={:0.2f}'.format(r)
        else:
            label = None

        if ax == None:
            if not 'linewidth' in argvs:
                plt.plot(x, y, linestyle='dashed', c=c, alpha=0.7, label=label, **argvs)
            else:
                plt.plot(x, y, linestyle='dashed', c=c, alpha=0.7, label=label, **argvs)
        else:
            if not 'linewidth' in argvs:
                ax.plot(x, y, linestyle='dashed', c=c, alpha=0.7, label=label, **argvs)
            else:
                ax.plot(x, y, linestyle='dashed', c=c, alpha=0.7, label=label, **argvs)
        plt.fill_between(x, y - err_list, y + err_list, alpha=0.5, color='gray',linewidth=0)

    def plot_pdp_shap_split_df_line_breakpoints_detail(self):

        fdir = join(self.this_class_arr, 'pdp_shap_split1')
        drt_list = global_drought_type_list

        for ELI in global_ELI_class_list:
            fdir_i = join(fdir, str(ELI))
            outdir_i = join(self.this_class_png, 'plot_pdp_shap_split_df_line_breakpoints_detail', str(ELI))
            T.mk_dir(outdir_i, force=True)
            for f in T.listdir(fdir_i):
                if not f.endswith('.df'):
                    continue
                var_name = f.split('.')[0].replace('shaply_', '')
                fpath = join(fdir_i, f)
                df = T.load_df(fpath)
                # T.print_head_n(df)
                start, end = Attribution_Dataframe().variables_threshold()[var_name]
                bins = np.linspace(start, end, 100)

                for drt in drt_list:
                    df_i = df[df['drought_type'] == drt]
                    df_group, bins_list_str = T.df_bin(df_i, var_name, bins)
                    # T.print_head_n(df)
                    x_mean_list = []
                    y_mean_list = []
                    y_err_list = []
                    for name, df_group_i in df_group:
                        x_i = name[0].left
                        # print(x_i)
                        # exit()
                        vals = df_group_i['shap_v'].tolist()

                        if len(vals) == 0:
                            continue
                        # mean = np.nanmean(vals)
                        mean = np.nanmedian(vals)
                        err = np.nanstd(vals)
                        y_mean_list.append(mean)
                        x_mean_list.append(x_i)
                        y_err_list.append(err)
                    y_mean_list = np.array(y_mean_list)
                    x_mean_list = np.array(x_mean_list)
                    # try:
                    #     y_mean_list = SMOOTH().mid_window_smooth(y_mean_list, window=5)
                    # except:
                    #     continue
                    # if not 'VPD' in var_name:
                    #     continue
                    o = self.change_point_rbeast(y_mean_list,
                                                season='none',
                                                tcp_minmax=(0, 1),
                                                    )
                    rb.plot(o, title=f'{ELI}_{var_name}_{drt}', fig=plt.figure(figsize=(15, 10)))
                    change_point = o.trend.cp[0]
                    change_point = int(change_point)
                    # exit()
                    x_vals = df_i[var_name].tolist()
                    y_vals = df_i['shap_v'].tolist()
                    color = global_drought_type_color_dict[drt]
                    # plt.show()

                    plt.savefig(join(outdir_i, f'{ELI}_{var_name}_{drt}.pdf'))
                    plt.close()

    def change_point_detection_ruptures(self,vals,n_bkps=2):
        import ruptures as rpt
        import pymannkendall as mk
        algo = rpt.Binseg(model="rbf").fit(vals)
        result = algo.predict(n_bkps=n_bkps)
        result = np.array(result) - 1

        return result

    def mann_kendall_change_point(self,data, window_size=5):
        import pymannkendall as mk
        change_points = []
        trends = []

        for i in range(0, len(data) - window_size + 1, window_size):
            window = data[i:i + window_size]
            test_result = mk.original_test(window)
            trends.append(test_result.trend)  # Capture trend for each window

            # Check if there is a trend and if there is a shift compared to the previous window
            if len(trends) > 1 and trends[-1] != trends[-2]:
                change_points.append(i + window_size)
        change_points = np.array(change_points) - 1

        return change_points, trends

    def change_point_detection_cusum(self,data, threshold=5., drift=0):
        # Cumulative sum
        S_pos, S_neg = np.zeros(len(data)), np.zeros(len(data))
        change_points = []

        for i in range(1, len(data)):
            S_pos[i] = max(0, S_pos[i - 1] + data[i] - data[i - 1] - drift)
            S_neg[i] = min(0, S_neg[i - 1] + data[i] - data[i - 1] + drift)

            if S_pos[i] > threshold or S_neg[i] < -threshold:
                change_points.append(i)
                S_pos[i], S_neg[i] = 0, 0  # Resetting after change point
        change_points = np.array(change_points)
        return change_points



    def detect_adf(self,data, p_value_threshold=0.05):
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(data)
        p_value = result[1]
        return p_value < p_value_threshold  # If p-value is below threshold, the series is stationary


    def detect_wavelet(self,vals):
        import numpy as np
        import pywt

        # Perform continuous wavelet transform (CWT)
        scales = np.arange(1, 128)
        coefficients, frequencies = pywt.cwt(vals, scales, 'mexh')
        return coefficients, frequencies

    def sliding_window_change_detection(self,data, window_size=10, threshold=0.2):
        change_points = []
        for i in range(len(data) - window_size):
            window_1 = data[i:i + window_size]
            window_2 = data[i + window_size:i + 2 * window_size]

            mean_diff = abs(np.mean(window_1) - np.mean(window_2))
            if mean_diff > threshold:
                change_points.append(i + window_size)
        return change_points

    def change_point_rbeast(self,vals,**kwargs):
        # beach, year = rb.load_example('googletrend')
        vals_df = pd.Series(vals)
        o = rb.beast(vals_df,quiet=True,start=0,**kwargs)
        # rb.plot(o,title=title,fig=plt.figure(figsize=(15,10)))
        return o
        pass

    def plot_pdp_shap_split_df_drought_mon(self):
        fdir = join(self.this_class_arr, 'pdp_shap_split1')
        drt_list = global_drought_type_list

        for ELI in global_ELI_class_list:
            fdir_i = join(fdir, str(ELI))
            outdir_i = join(self.this_class_png, 'pdp_shap_split_df_scatter1', str(ELI))
            T.mk_dir(outdir_i, force=True)
            for f in T.listdir(fdir_i):
                if not f.endswith('.df'):
                    continue
                var_name = f.split('.')[0].replace('shaply_', '')
                if not var_name == 'drought_mon':
                    continue
                fpath = join(fdir_i, f)
                df = T.load_df(fpath)
                # T.print_head_n(df)
                start, end = Attribution_Dataframe().variables_threshold()[var_name]
                bins = np.linspace(start, end, 50)

                for drt in drt_list:
                    df_i = df[df['drought_type'] == drt]
                    df_group, bins_list_str = T.df_bin(df_i, var_name, bins)
                    x_unique = df_i[var_name].unique()
                    x_unique = list(x_unique)
                    x_unique.sort()
                    y_vals_list = []
                    for x_i in x_unique:
                        y_i = df_i[df_i[var_name] == x_i]['shap_v'].tolist()
                        y_vals_list.append(y_i)
                    plt.boxplot(y_vals_list, positions=x_unique, showfliers=False, showmeans=False)
                    # plt.show()
                    plt.title(f'{ELI}\n{var_name}\n{drt}')
                    # plt.xlim(start, end)
                    plt.ylim(-0.6, 0.6)
                    # plt.show()

                    plt.savefig(join(outdir_i, f'{var_name}_{drt}.pdf'))
                    plt.close()

    def plot_importances(self):
        ELI_class_list = global_ELI_class_list
        outdir = join(self.this_class_png, 'importances')
        T.mk_dir(outdir, force=True)
        # mode = 'xgboost'
        mode = 'shap'

        for ELI in ELI_class_list:
            plt.figure()
            if mode == 'xgboost':
                imp_dict_fpath = join(self.this_class_arr,'pdp_shap',str(ELI), 'imp_xgboost.npy')
            elif mode == 'shap':
                imp_dict_fpath = join(self.this_class_arr,'pdp_shap',str(ELI), 'shaply_imp_dict.npy')
            else:
                raise

            imp_dict = T.load_npy(imp_dict_fpath)
            imp_dict = T.sort_dict_by_value(imp_dict,descending=False)

            x_list = []
            y_list = []
            for key in imp_dict.keys():
                x_list.append(key)
                y_list.append(imp_dict[key])

            plt.barh(x_list, y_list)
            print(x_list)
            # plt.title(f'R2_{R2}')
            plt.xticks(rotation=90)
            plt.title(f'{ELI}_{mode}')

            plt.tight_layout()
            outf = join(outdir,f'{ELI}_{mode}.pdf')
            plt.savefig(outf)
            plt.close()
        # plt.show()
        T.open_path_and_file(outdir)

        pass

    def sort_dict_by_value(self, input_dict,descending=True):
        return dict(sorted(input_dict.items(), key=lambda item: item[1], reverse=descending))

    def __feature_importances_shap_values(self,shap_values, features):
        '''
        Prints the feature importances based on SHAP values in an ordered way
        shap_values -> The SHAP values calculated from a shap.Explainer object
        features -> The name of the features, on the order presented to the explainer
        '''
        # Calculates the feature importance (mean absolute shap value) for each feature
        importances = []
        for i in range(shap_values.values.shape[1]):
            importances.append(np.mean(np.abs(shap_values.values[:, i])))
        # Calculates the normalized version
        # importances_norm = softmax(importances)
        # Organize the importances and columns in a dictionary
        feature_importances = {fea: imp for imp, fea in zip(importances, features)}
        # feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
        # Sorts the dictionary
        feature_importances = {k: v for k, v in
                               sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)}
        # feature_importances_norm = {k: v for k, v in
        #                             sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse=True)}
        # Prints the feature importances
        # for k, v in feature_importances.items():
        #     print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")

        return feature_importances
        # return feature_importances_norm


    def __train_model(self,X,y):
        from sklearn.model_selection import train_test_split
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=1, test_size=0.3) # split the data into training and testing
        # model = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=7,) # build a random forest model
        # rf.fit(X_train, y_train) # train the model
        # r2 = rf.score(X_test,y_test)
        model = xgb.XGBRegressor(objective="reg:squarederror",booster='gbtree',n_estimators=100,
                                 max_depth=10,eta=0.005,random_state=1,n_jobs=24)
        # model = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=12)
        model.fit(X_train, y_train)
        # model.fit(X_train, y_train)
        # Get predictions
        y_pred = model.predict(X_test)
        # plt.scatter(y_test, y_pred)
        # plt.show()
        r = stats.pearsonr(y_test, y_pred)
        r2 = r[0] ** 2
        print('r2:', r2)
        # exit()

        return model,y,y_pred

    def __variables_ranges(self):
        range_dict = {
            'Energy-Limited':{
                'SOS': (-20, 20),
                'VPD-anomaly': (-1.5, 2),
                'Temperature-anomaly_detrend': (-1.5, 1.5),
                'Temperature-anomaly': (-1.5, 1.5),
                'FAPAR-anomaly_detrend': (-3, 3),
                'Radiation-anomaly': (-1.5, 2),
                'delta_Topt_T': (-8, 10),
                'drought_mon': (5, 10),
                'NDVI-anomaly_detrend': (-3, 3),
                'NDVI-anomaly_with_trend': (-3, 3),
                'GLEAM-SMRoot-anomaly_detrend': (-3, 2),
                # 'SILT': (0, 60),
                'sand': (0, 1000),
                'Temperature_quantile': (0, 100),
            },
            'Water-Limited':{
                'SOS': (-20, 20),
                'VPD-anomaly': (-1,2),
                'Temperature-anomaly_detrend': (-1,1.5),
                'Temperature-anomaly': (-1,1.5),
                'FAPAR-anomaly_detrend': (-3, 3),
                'Radiation-anomaly': (-1.5,1.5),
                'delta_Topt_T': (-6, 10),
                'drought_mon': (5, 10),
                'NDVI-anomaly_detrend': (-3, 3),
                'NDVI-anomaly_with_trend': (-3, 3),
                'GLEAM-SMRoot-anomaly_detrend': (-2, 2),
                'SILT': (0, 1000),
                'sand': (0, 1000),
                'Temperature_quantile': (0, 100),
            }
        }

        return range_dict

class SEM:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('SEM', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe/Dataframe.df')
        pass

    def run(self):
        import statistic
        # self.copy_df()
        df = self.__gen_df_init()
        df = self.filter_df(df)
        # add NDVI anomaly
        # NDVI_data_obj = Load_Data().NDVI_anomaly_detrend
        # df = self.add_variables(df,NDVI_data_obj)

        # add VPD anomaly
        # VPD_data_obj = Load_Data().VPD_anomaly
        # df = self.add_variables(df,VPD_data_obj)

        # add T anomaly
        # T_data_obj = Load_Data().Temperature_anomaly
        # df = self.add_variables(df,T_data_obj)

        # add T origin
        # T_data_obj = Load_Data().Temperature_origin
        # df = self.add_variables(df,T_data_obj)

        # add srad
        # srad_data_obj = Load_Data().Srad_anomaly
        # df = self.add_variables(df,srad_data_obj)

        # add ET
        # ET_data_obj = Load_Data().GLEAM_Et_anomaly
        # df = self.add_variables(df,ET_data_obj)

        # add spring phenology
        # df = self.add_SOS(df)

        # add optimal temperature
        # df = self.add_optimal_temperature(df)
        # df = self.add_delta_optimal_temperature(df)

        # df = statistic.Dataframe_func(df).df

        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)

        # self.pair_plot(df)
        # self.build_model(df)


        # self.check_variables(df)
        # exit()

        pass

    def __gen_df_init(self):
        if not os.path.isfile(self.dff):
            df = pd.DataFrame()
            T.save_df(df,self.dff)
            return df
        else:
            df,dff = self.__load_df()
            return df

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print('len(df):',len(df))
        return df,dff

    def filter_df(self,df):
        df = df[df['drought_scale']=='spi03']
        return df

    def copy_df(self):
        import analysis
        if isfile(self.dff):
            print('already exists: ', self.dff)
            print('press enter to overwrite')
            pause()
            pause()
            pause()
        T.mkdir(join(self.this_class_arr, 'dataframe'))
        dff = join(analysis.Pick_Drought_Events().this_class_arr, 'drought_dataframe/drought_dataframe.df')
        df = T.load_df(dff)
        T.save_df(df,self.dff)
        T.df_to_excel(df, self.dff)

    def add_variables(self,df,data_obj):
        # data_dict,var_name,valid_range = Load_Data().NDVI_anomaly_detrend()
        data_dict,var_name,valid_range = data_obj()

        vals_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            year = row['drought_year']
            if not pix in data_dict:
                vals_list.append(np.nan)
                continue
            vals = data_dict[pix]
            vals = np.array(vals)
            if np.nanstd(vals) == 0:
                vals_list.append(np.nan)
                continue

            vals[vals<valid_range[0]] = np.nan
            vals[vals>valid_range[1]] = np.nan
            # vals_reshape = np.reshape(vals,(-1,12))
            vals_gs = T.monthly_vals_to_annual_val(vals,grow_season=global_gs)
            val_drought_year = vals_gs[year-global_start_year]
            vals_list.append(val_drought_year)

        df[var_name] = vals_list
        return df

    def add_SOS(self,df):
        import analysis
        pheno_str = 'early_start'
        fpath = join(analysis.Phenology().this_class_arr, 'phenology_df/phenology_df.df')
        phenology_df = T.load_df(fpath)
        cols = list(phenology_df.columns)
        print(cols)
        pheno_spatial_dict = {}
        for i, row in phenology_df.iterrows():
            pix = row['pix']
            early_start = row[pheno_str]
            early_start_dict = dict(early_start)
            phenology_anomaly_dict = self.phenology_anomaly_dict(early_start_dict)
            pheno_spatial_dict[pix] = phenology_anomaly_dict
        pheno_val_list = []
        for i,row in df.iterrows():
            pix = row['pix']
            year = row['drought_year']
            if not pix in pheno_spatial_dict:
                pheno_val_list.append(np.nan)
                continue
            if not year in pheno_spatial_dict[pix]:
                pheno_val_list.append(np.nan)
                continue
            pheno_val = pheno_spatial_dict[pix][year]
            if pheno_val > 20:
                pheno_val_list.append(np.nan)
                continue
            if pheno_val < -20:
                pheno_val_list.append(np.nan)
                continue
            pheno_val_list.append(pheno_val)
        df['SOS_anomaly'] = pheno_val_list
        return df

    def phenology_anomaly_dict(self,phenology_dict):
        vals = list(phenology_dict.values())
        mean = np.nanmean(vals)
        phenology_anomaly_dict = {}
        for year in phenology_dict:
            val = phenology_dict[year]
            anomaly = val - mean
            phenology_anomaly_dict[year] = anomaly
        return phenology_anomaly_dict

    def add_optimal_temperature(self,df):
        import analysis
        Topt_f = join(analysis.Optimal_temperature().this_class_tif,'optimal_temperature/LT_Baseline_NT_origin_step_0.5_celsius_resample.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(Topt_f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'Topt')
        # T.print_head_n(df)
        return df

    def add_delta_optimal_temperature(self,df):
        df['delta_optimal_temp'] = df['Temperature-origin'] - df['Topt']
        return df


    def check_variables(self,df):
        # col_name = 'NDVI-anomaly_detrend'
        col_name = 'SOS_anomaly'
        df = df.dropna(subset=[col_name])
        DIC_and_TIF().plot_df_spatial_pix(df,global_land_tif)
        plt.show()
        print(len(df))
        exit()
        pass

    def pair_plot(self,df):
        outdir = join(self.this_class_png,'pair_plot')
        T.mk_dir(outdir)
        cols = [
            'NDVI-anomaly_detrend',
            'SOS_anomaly',
            'Topt',
            'VPD-anomaly',
            'GLEAM-Et-anomaly',
            'Radiation-anomaly',
        ]
        df = df.dropna(subset=cols)
        sns.pairplot(df,vars=cols,kind='hist')
        outf = join(outdir,'pair_plot.png')
        plt.savefig(outf,dpi=300)
        plt.close()
        pass

    def model_description(self):
        desc = '''
        # regression
        NDVI_anomaly_detrend ~ SOS_anomaly + delta_optimal_temp + VPD_anomaly + GLEAM_Et_anomaly + Radiation_anomaly + Temperature_anomaly
        SOS_anomaly ~ Radiation_anomaly + Temperature_anomaly
        GLEAM_Et_anomaly ~ VPD_anomaly + SOS_anomaly + Radiation_anomaly + Temperature_anomaly
        # residual correlations
        '''
        a='''
        SOS_anomaly ~ Topt + Temperature-anomaly
        GLEAM-Et-anomaly ~ VPD-anomaly + SOS_anomaly + Radiation-anomaly + Temperature-anomaly
        '''
        return desc

    def build_model(self, df):
        # exit()
        drought_type_list = global_drought_type_list
        AI_class_list = global_AI_class_list
        season_list = global_drought_season_list
        cols = df.columns
        for col in cols:
            col_new = col.replace('-', '_')
            df[col_new] = df[col]
        for season in season_list:
            for drt in drought_type_list:
                for AI_class in AI_class_list:
                    # df_season = df[df['drought_season'] == season]
                    df_season = df
                    df_drt = df_season[df_season['drought_type'] == drt]
                    df_AI = df_drt[df_drt['AI_class'] == AI_class]

                    outdir = join(self.this_class_png,'model')
                    T.mk_dir(outdir)
                    # outf = join(outdir, f'{drt}_{AI_class}_{season}')
                    outf = join(outdir, f'{drt}_{AI_class}')
                    T.mk_dir(outf)
                    desc = self.model_description()
                    mod = semopy.Model(desc)
                    res = mod.fit(df_AI)
                    semopy.report(mod, outf)

def copy_files():
    f = join(this_root,"conf\land_reproj.tif")
    print(isfile(f))
    dest_f = join(this_root,"conf\land_reproj_copy.tif")
    shutil.copyfile(f,dest_f)
    pass



def main():
    # SEM().run()
    # MAT_Topt().run()
    # MAT_Topt1().run()
    # Attribution_Dataframe().run()
    # Random_forests().run()
    SHAP().run()
    # copy_files()
    pass

if __name__ == '__main__':
    main()