# coding=utf-8
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
        # T.save_df(df, self.dff)
        # T.df_to_excel(df, self.dff)

        # self.pair_plot(df)
        self.build_model(df)


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
        NDVI_anomaly_detrend ~ SOS_anomaly + Topt + VPD_anomaly + GLEAM_Et_anomaly + Radiation_anomaly + Temperature_anomaly
        SOS_anomaly ~ Topt + Temperature_anomaly
        GLEAM_Et_anomaly ~ VPD_anomaly + SOS_anomaly + Radiation_anomaly + Temperature_anomaly
        # residual correlations
        '''
        a='''
        SOS_anomaly ~ Topt + Temperature-anomaly
        GLEAM-Et-anomaly ~ VPD-anomaly + SOS_anomaly + Radiation-anomaly + Temperature-anomaly
        '''
        return desc

    def build_model(self, df):
        cols = df.columns
        for col in cols:
            col_new = col.replace('-','_')
            df[col_new] = df[col]
        # print(df.columns)
        # exit()

        outdir = join(self.this_class_png,'model')
        T.mk_dir(outdir)
        outf = join(outdir, 'SEM_report')
        T.mk_dir(outf)
        desc = self.model_description()
        mod = semopy.Model(desc)
        res = mod.fit(df)
        semopy.report(mod, outf)

def main():
    SEM().run()
    pass

if __name__ == '__main__':
    main()