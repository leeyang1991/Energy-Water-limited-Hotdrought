# coding=utf-8

from meta_info import *
result_root_this_script = join(results_root, 'Supplementary')

class NDVI_3g_analysis:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'NDVI_3g_analysis',
            result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr,'dataframe.df')
        pass

    def run(self):
        # self.copy_df()
        # ----------------------------
        df = self.__gen_df_init()
        # df = self.add_GS_NDVI3g(df)

        # T.save_df(df,self.dff)
        # T.df_to_excel(df,self.dff)
        # ----------------------------

        self.tif_drought_year(df)
        self.plot_tif_drought_year()

        pass

    def copy_df(self):
        print('Warning: this function will overwrite the dataframe')
        print('Warning: this function will overwrite the dataframe')
        print('Warning: this function will overwrite the dataframe')
        pause()
        pause()
        import statistic
        dff = statistic.Dataframe().dff
        df = T.load_df(dff)
        df = df[df['drought_scale'] == 'spi03']
        T.save_df(df,self.dff)
        T.df_to_excel(df,self.dff)

    def add_GS_NDVI3g(self,df):
        spatial_dict,var_name = Load_Data().NDVI_3g_anomaly_detrend()
        year_list = year_range_str_to_list(global_VIs_year_range_dict['NDVI3g'])
        # print(year_list)
        # exit()
        gs_mon_index = np.array(global_gs,dtype=int) - 1
        gs_vals_list = []
        for i,row in tqdm(df.iterrows(),total=len(df),desc='add_GS_NDVI'):
            pix = row['pix']
            drought_year = row['drought_year']
            if not drought_year in year_list:
                gs_vals_list.append(np.nan)
                continue
            vals = spatial_dict[pix]
            vals_reshape = np.array(vals).reshape(-1,12)
            vals_reshape_dict = dict(zip(year_list,vals_reshape))
            drought_year_vals = vals_reshape_dict[drought_year]
            gs_vals = drought_year_vals[gs_mon_index]
            gs_vals_mean = np.nanmean(gs_vals)
            gs_vals_list.append(gs_vals_mean)
        df[var_name+'_GS'] = gs_vals_list
        df = df[df[var_name+'_GS']!=0]

        return df

    def tif_drought_year(self,df):
        outdir = join(self.this_class_tif,'drought_year')
        T.mk_dir(outdir)
        drought_type_list = global_drought_type_list

        for drt in drought_type_list:
            df_drt = df[df['drought_type']==drt]
            spatial_dict = {}
            df_group_dict = T.df_groupby(df_drt,'pix')
            for pix in df_group_dict:
                df_pix = df_group_dict[pix]
                drought_year_value = df_pix['NDVI3g-anomaly_detrend_GS'].tolist()
                drought_year_mean = np.nanmean(drought_year_value)
                spatial_dict[pix] = drought_year_mean
            outf = join(outdir,f'{drt}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict,outf)

    def plot_tif_drought_year(self):
        fdir = join(self.this_class_tif,'drought_year')
        outdir = join(self.this_class_png,'drought_year')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            outf = join(outdir,f.replace('.tif','.png'))
            m,ret = Plot().plot_ortho(fpath,vmin=-1,vmax=1,cmap='RdBu')
            plt.savefig(outf, dpi=300)
            plt.close()
        T.open_path_and_file(outdir)
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


class CSIF_analysis:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'CSIF_analysis',
            result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr,'dataframe.df')
        pass

    def run(self):
        # self.copy_df()
        # ----------------------------
        df = self.__gen_df_init()
        # df = self.add_GS_CSIF(df)
        # T.save_df(df,self.dff)
        # T.df_to_excel(df,self.dff)
        # ----------------------------
        self.tif_drought_year(df)
        self.plot_tif_drought_year()
        pass

    def copy_df(self):
        print('Warning: this function will overwrite the dataframe')
        print('Warning: this function will overwrite the dataframe')
        print('Warning: this function will overwrite the dataframe')
        pause()
        pause()
        import statistic
        dff = statistic.Dataframe().dff
        df = T.load_df(dff)
        df = df[df['drought_scale'] == 'spi03']
        T.save_df(df,self.dff)
        T.df_to_excel(df,self.dff)

    def add_GS_CSIF(self,df):
        spatial_dict,var_name = Load_Data().CSIF_anomaly_detrend()
        year_list = year_range_str_to_list(global_VIs_year_range_dict['CSIF'])
        # print(year_list)
        # exit()
        gs_mon_index = np.array(global_gs,dtype=int) - 1
        gs_vals_list = []
        for i,row in tqdm(df.iterrows(),total=len(df),desc='add_GS_CSIF'):
            pix = row['pix']
            drought_year = row['drought_year']
            if not drought_year in year_list:
                gs_vals_list.append(np.nan)
                continue
            vals = spatial_dict[pix]
            vals_reshape = np.array(vals).reshape(-1,12)
            vals_reshape_dict = dict(zip(year_list,vals_reshape))
            drought_year_vals = vals_reshape_dict[drought_year]
            gs_vals = drought_year_vals[gs_mon_index]
            gs_vals_mean = np.nanmean(gs_vals)
            gs_vals_list.append(gs_vals_mean)
        df[var_name+'_GS'] = gs_vals_list
        df = df[df[var_name+'_GS']!=0]

        return df

    def tif_drought_year(self,df):
        outdir = join(self.this_class_tif,'drought_year')
        T.mk_dir(outdir)
        drought_type_list = global_drought_type_list

        for drt in drought_type_list:
            df_drt = df[df['drought_type']==drt]
            spatial_dict = {}
            df_group_dict = T.df_groupby(df_drt,'pix')
            for pix in df_group_dict:
                df_pix = df_group_dict[pix]
                drought_year_value = df_pix['CSIF-anomaly_detrend_GS'].tolist()
                drought_year_mean = np.nanmean(drought_year_value)
                spatial_dict[pix] = drought_year_mean
            outf = join(outdir,f'{drt}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict,outf)

    def plot_tif_drought_year(self):
        fdir = join(self.this_class_tif,'drought_year')
        outdir = join(self.this_class_png,'drought_year')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            outf = join(outdir,f.replace('.tif','.png'))
            m,ret = Plot().plot_ortho(fpath,vmin=-1,vmax=1,cmap='RdBu')
            plt.savefig(outf, dpi=300)
            plt.close()
        T.open_path_and_file(outdir)
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

class Multi_SPI_scale_analysis:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Multi_SPI_scale_analysis',
            result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe.df')
        self.spi_scale_list = ['spi03','spi06','spi09','spi12']
        pass

    def run(self):
        # self.copy_df()
        # ----------------------------
        # df = self.__gen_df_init()
        # df = self.add_GS_NDVI4g(df)
        # T.save_df(df,self.dff)
        # T.df_to_excel(df,self.dff)

        # self.tif_drought_year(df)
        self.plot_tif_drought_year()
        pass

    def copy_df(self):
        print('Warning: this function will overwrite the dataframe')
        print('Warning: this function will overwrite the dataframe')
        print('Warning: this function will overwrite the dataframe')
        pause()
        pause()
        import statistic
        dff = statistic.Dataframe().dff
        df = T.load_df(dff)
        selected_scale_list = self.spi_scale_list
        df = df[df['drought_scale'].isin(selected_scale_list)]
        T.save_df(df,self.dff)
        T.df_to_excel(df,self.dff)

    def add_GS_NDVI4g(self,df):
        spatial_dict,var_name = Load_Data().NDVI_anomaly_detrend()
        year_list = year_range_str_to_list(global_VIs_year_range_dict['NDVI4g'])
        # print(year_list)
        # exit()
        gs_mon_index = np.array(global_gs,dtype=int) - 1
        gs_vals_list = []
        for i,row in tqdm(df.iterrows(),total=len(df),desc='add_GS_NDVI'):
            pix = row['pix']
            drought_year = row['drought_year']
            if not drought_year in year_list:
                gs_vals_list.append(np.nan)
                continue
            vals = spatial_dict[pix]
            vals_reshape = np.array(vals).reshape(-1,12)
            vals_reshape_dict = dict(zip(year_list,vals_reshape))
            drought_year_vals = vals_reshape_dict[drought_year]
            gs_vals = drought_year_vals[gs_mon_index]
            gs_vals_mean = np.nanmean(gs_vals)
            gs_vals_list.append(gs_vals_mean)
        df[var_name+'_GS'] = gs_vals_list
        df = df[df[var_name+'_GS']!=0]

        return df

    def tif_drought_year(self,df):
        outdir = join(self.this_class_tif,'drought_year')
        T.mk_dir(outdir)
        drought_type_list = global_drought_type_list
        spi_scale_list = self.spi_scale_list
        for scale in spi_scale_list:
            df_scale = df[df['drought_scale']==scale]
            for drt in drought_type_list:
                df_drt = df_scale[df_scale['drought_type']==drt]
                spatial_dict = {}
                df_group_dict = T.df_groupby(df_drt,'pix')
                for pix in tqdm(df_group_dict,desc=f'{scale}_{drt}'):
                    df_pix = df_group_dict[pix]
                    drought_year_value = df_pix['NDVI-anomaly_detrend_GS'].tolist()
                    drought_year_mean = np.nanmean(drought_year_value)
                    spatial_dict[pix] = drought_year_mean
                outf = join(outdir,f'{scale}_{drt}.tif')
                DIC_and_TIF().pix_dic_to_tif(spatial_dict,outf)

    def plot_tif_drought_year(self):
        fdir = join(self.this_class_tif,'drought_year')
        outdir = join(self.this_class_png,'drought_year')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            outf = join(outdir,f.replace('.tif','.png'))
            m,ret = Plot().plot_ortho(fpath,vmin=-1,vmax=1,cmap='RdBu')
            plt.title(f.replace('.tif',''))
            plt.savefig(outf, dpi=300)
            plt.close()
        T.open_path_and_file(outdir)
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



def main():
    # NDVI_3g_analysis().run()
    # CSIF_analysis().run()
    Multi_SPI_scale_analysis().run()
    pass

if __name__ == '__main__':
    main()