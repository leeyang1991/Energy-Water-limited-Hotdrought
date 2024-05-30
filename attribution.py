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
        self.temperature_during_drought()
        # self.build_df()
        # self.foo()
        # self.foo2()
        pass


    def build_df(self):
        import analysis
        import statistic
        # Topt_f = join(analysis.Optimal_temperature().this_class_tif,'optimal_temperature/LT_Baseline_NT_origin_step_0.5_celsius_resample.tif')
        Topt_f = join(analysis.Optimal_temperature().this_class_tif,'optimal_temperature/TCSIF-optimal_temperature.tif')
        # MAT_f = join(data_root,r"CRU_tmp\mat\mat_gs.tif")
        MAT_f = join(data_root,r"CRU_tmp\max_annual_temperature\max_annual_temperature_gs.tif")
        Temp_during_drought = join(self.this_class_tif,'Temperature_during_drought/max_Temperature_during_drought.tif')

        compensation_excerbation_f = join(statistic.Compensation_Excerbation().this_class_tif,r"delta_hot_normal\drought_year_1.tif")

        Topt_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(Topt_f)
        MAT_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(MAT_f)
        Temp_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(Temp_during_drought)
        compensation_excerbation_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(compensation_excerbation_f)

        all_dict = {}
        all_dict['Topt'] = Topt_spatial_dict
        all_dict['MAT'] = MAT_spatial_dict
        all_dict['Temp_during_drought'] = Temp_spatial_dict
        all_dict['compensation_excerbation'] = compensation_excerbation_spatial_dict

        df = T.spatial_dics_to_df(all_dict)
        df = df.dropna(how='any')

        outf = join(self.this_class_arr, 'dataframe/Dataframe.df')
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
        import statistic
        compensation_excerbation_tif = join(statistic.Compensation_Excerbation().this_class_tif,'delta_hot_normal/drought_year_1.tif')
        compensation_excerbation_arr = DIC_and_TIF().spatial_tif_to_arr(compensation_excerbation_tif)
        Topt_MAT_delta_tif = join(self.this_class_tif,'mat_Topt_delta/Topt_MAT_delta.tif')
        Topt_MAT_delta_arr = DIC_and_TIF().spatial_tif_to_arr(Topt_MAT_delta_tif)

        compensation_excerbation_arr_flatten = compensation_excerbation_arr.flatten()
        Topt_MAT_delta_arr_flatten = Topt_MAT_delta_arr.flatten()
        df = pd.DataFrame()
        df['compensation_excerbation_arr_flatten'] = compensation_excerbation_arr_flatten
        df['Topt_MAT_delta_arr_flatten'] = Topt_MAT_delta_arr_flatten
        df = df.dropna(how='any')
        T.print_head_n(df)
        bins = np.linspace(-8,8,51)
        df_group, bins_list_str = T.df_bin(df,'Topt_MAT_delta_arr_flatten',bins)

        x_list = []
        y_list = []
        err_list = []
        for name, df_group_i in df_group:
            vals = df_group_i['compensation_excerbation_arr_flatten'].tolist()
            mean = np.nanmean(vals)
            err,_,_ = T.uncertainty_err(vals)
            x = name[0].left
            x_list.append(x)
            y_list.append(mean)
            err_list.append(err)
        plt.hist(df['Topt_MAT_delta_arr_flatten'], bins=100, range=(-8, 8), zorder=-99,color='gray',alpha=0.5)
        # plt.hist(df['Topt_MAT_delta_arr_flatten'], bins=100, zorder=-99,color='gray',alpha=0.5)
        plt.xlabel('MAT - Topt')
        plt.ylabel('Compensation Excerbation')
        plt.twinx()
        plt.plot(x_list, y_list, c='r')
        plt.fill_between(x_list, np.array(y_list) - np.array(err_list), np.array(y_list) + np.array(err_list),
                         alpha=0.3)

        plt.show()
        exit()
        pass

    def temperature_during_drought(self):
        outdir = join(self.this_class_tif,'temperature_during_drought')
        T.mk_dir(outdir)
        dff = SEM().dff
        df = T.load_df(dff)
        # df = df[df['drought_type'] == 'hot-drought']
        df = self.add_variables(df,Load_Data().Temperature_origin)
        T.print_head_n(df)
        exit()
        df_pix_dict = T.df_groupby(df, 'pix')
        Temperature_origin_mean_dict = {}
        for pix in df_pix_dict:
            df_pix = df_pix_dict[pix]
            df_pix = T.df_drop_duplicates(df_pix, 'drought_year')
            Temperature_origin = df_pix['max_Temperature-origin'].tolist()
            Temperature_origin_mean = np.nanmean(Temperature_origin)
            Temperature_origin_mean_dict[pix] = Temperature_origin_mean
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(Temperature_origin_mean_dict)
        outf = join(outdir,'max_Temperature_during_hot-drought.tif')
        DIC_and_TIF().arr_to_tif(arr,outf)

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
            vals_gs = T.monthly_vals_to_annual_val(vals,grow_season=global_gs,method='max')
            # vals_gs = T.monthly_vals_to_annual_val(vals,grow_season=global_gs,method='mean')
            val_drought_year = vals_gs[year-global_start_year]
            vals_list.append(val_drought_year)

        df['max_'+var_name] = vals_list
        return df

    def foo(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        # exit()
        # print('len(df):',len(df))
        delta_Tempdr_vs_Topt = df['Temp_during_drought'] - df['Topt']
        df['delta_Tempdr_vs_Topt'] = delta_Tempdr_vs_Topt
        bins = np.linspace(5,30,51)
        df_group, bins_list_str = T.df_bin(df,'MAT',bins)
        x_list = []
        y_list = []
        y1_list = []
        err_list = []
        for name,df_group_i in df_group:
            vals = df_group_i['delta_Tempdr_vs_Topt'].tolist()
            vals1 = df_group_i['compensation_excerbation'].tolist()
            mean = np.nanmean(vals)
            mean1 = np.nanmean(vals1)
            # err,_,_ = T.uncertainty_err(vals)
            err = np.nanstd(vals)
            x = name[0].left
            x_list.append(x)
            y_list.append(mean)
            y1_list.append(mean1)
            err_list.append(err)
        plt.plot(x_list,y_list)
        plt.fill_between(x_list, np.array(y_list) - np.array(err_list), np.array(y_list) + np.array(err_list),
                         alpha=0.3)
        plt.ylabel('delta_Tempdr_vs_Topt')
        plt.xlabel('MAT')
        plt.twinx()
        plt.plot(x_list,y1_list,c='r')

        # plt.ylim(-3,3)
        plt.show()
        pass

    def foo2(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        # exit()
        # print('len(df):',len(df))
        bin_cols_list = ['Temp_during_drought','Topt','MAT']
        bins = np.linspace(5,28,24)
        for col in bin_cols_list:
            df_group, bins_list_str = T.df_bin(df,col,bins)
            x_list = []
            y_list = []
            err_list = []
            for name,df_group_i in df_group:
                vals = df_group_i['compensation_excerbation'].tolist()
                mean = np.nanmean(vals)
                err,_,_ = T.uncertainty_err(vals)
                # err = np.nanstd(vals)
                x = name[0].left
                x_list.append(x)
                y_list.append(mean)
                err_list.append(err)
            # plt.figure()
            plt.plot(x_list,y_list,label=col)
            plt.fill_between(x_list, np.array(y_list) - np.array(err_list), np.array(y_list) + np.array(err_list),
                             alpha=0.3)
            plt.ylabel('compensation_excerbation')
            # plt.xlabel(col)

        # plt.ylim(-3,3)
        plt.hlines(0,5,28,'k',linestyles='dashed')
        plt.legend()
        plt.show()
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