# coding=utf-8
import shutil

import matplotlib.pyplot as plt
import semopy
from meta_info import *
result_root_this_script = join(results_root, 'attribution')


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
        self.compensation_excerbation_delta()
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
        # MAT_f = join(data_root,r"CRU_tmp\mat\mat_gs.tif")
        MAT_f = join(data_root,r"CRU_tmp\max_annual_temperature\max_annual_temperature_gs.tif")
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
        plt.savefig(outf)
        # plt.show()

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


def copy_files():
    f = join(this_root,"conf\land_reproj.tif")
    print(isfile(f))
    dest_f = join(this_root,"conf\land_reproj_copy.tif")
    shutil.copyfile(f,dest_f)
    pass

def main():
    # SEM().run()
    MAT_Topt().run()
    # copy_files()
    pass

if __name__ == '__main__':
    main()