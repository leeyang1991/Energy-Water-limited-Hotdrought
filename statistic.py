# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from meta_info import *
result_root_this_script = join(results_root, 'statistic')

class Dataframe_func:

    def __init__(self,df,is_clean_df=True):
        print('add lon lat')
        df = self.add_lon_lat(df)

        print('add NDVI mask')
        df = self.add_NDVI_mask(df)
        if is_clean_df == True:
            df = self.clean_df(df)

        # print('add landcover')
        # df = self.add_GLC_landcover_data_to_df(df)

        print('add Aridity Index')
        df = self.add_AI_to_df(df)

        print('add ELI')
        df = self.add_ELI_to_df(df)

        print('add ELI_reclass')
        df = self.ELI_reclass(df)

        print('add AI_reclass')
        df = self.AI_reclass(df)
        # print('add ELI_significance')
        # df = self.add_ELI_significance(df)
        # print('add koppen')
        # df = self.add_koppen(df)

        self.df = df

    def clean_df(self,df):

        df = df[df['lat']>=30]
        # df = df[df['landcover_GLC'] != 'Crop']
        df = df[df['NDVI_MASK'] == 1]
        # df = df[df['ELI_significance'] == 1]
        return df

    def add_GLC_landcover_data_to_df(self, df):
        f = join(data_root,'GLC2000/reclass_lc_dic.npy')
        val_dic=T.load_npy(f)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            val_list.append(vals)
        df['landcover_GLC'] = val_list
        return df

    def add_NDVI_mask(self,df):
        # f =rf'C:/Users/pcadmin/Desktop/Data/Base_data/NDVI_mask.tif'
        f = join(data_root, 'NDVI4g/NDVI_mask.tif')
        print(f)

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'NDVI_MASK'
        print(f_name)
        # exit()
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            if vals < -99:
                val_list.append(np.nan)
                continue
            val_list.append(vals)
        df[f_name] = val_list
        return df

    def add_AI_to_df(self, df):
        f = join(data_root, 'Aridity_Index/aridity_index.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'aridity_index')
        return df

    def add_lon_lat(self,df):
        df = T.add_lon_lat_to_df(df, DIC_and_TIF())
        return df

    def add_ELI_to_df(self,df):
        import analysis
        f = join(analysis.Water_energy_limited_area().this_class_tif, 'ELI/GLEAM-ET_ERA-SM_Temperature.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'ELI')
        return df

    def ELI_reclass(self,df):
        ELI_class = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            ELI = row['ELI']
            if ELI < 0:
                ELI_class.append('Energy-Limited')
            elif ELI > 0:
                ELI_class.append('Water-Limited')
            else:
                ELI_class.append(np.nan)
        df['ELI_class'] = ELI_class
        return df

    def AI_reclass(self,df):
        AI_class = []
        for i,row in df.iterrows():
            AI = row['aridity_index']
            if AI < 0.65:
                AI_class.append('Arid')
            elif AI >= 0.65:
                AI_class.append('Humid')
            elif np.isnan(AI):
                AI_class.append(np.nan)
            else:
                print(AI)
                raise ValueError('AI error')
        df['AI_class'] = AI_class
        return df

    def add_koppen(self,df):
        f = join(data_root, 'koppen/koppen_reclass_dic.npy')
        val_dic = T.load_npy(f)
        df = T.add_spatial_dic_to_df(df, val_dic, 'Koppen')
        return df

    # def add_ELI_significance(self,df):
    #     from Chapter5 import analysis
    #     f = join(Water_energy_limited_area().this_class_tif, 'significant_pix_r/ELI_Temp_significance.tif')
    #     spatial_dict = DIC_and_TIF().spatial_tif_to_dic(f)
    #     df = T.add_spatial_dic_to_df(df, spatial_dict, 'ELI_significance')
    #
    #     return df

class Dataframe:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Dataframe', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe.df')
        pass

    def run(self):
        self.copy_df()
        df = self.__gen_df_init()
        df = Dataframe_func(df).df

        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)
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

    def copy_df(self):
        print('Warning: this function will overwrite the dataframe')
        print('Warning: this function will overwrite the dataframe')
        print('Warning: this function will overwrite the dataframe')
        pause()
        pause()
        import analysis
        dff = join(analysis.Pick_Drought_Events().this_class_arr,'drought_dataframe/drought_dataframe.df')
        df = T.load_df(dff)
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)


class Compensation_Excerbation:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Compensation_Excerbation', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'Compensation_Excerbation.df')
        pass

    def run(self):
        # df = self.__gen_df_init()
        # T.print_head_n(df)
        # exit()
        # calculate
        # df = self.identification_two_modes(df)
        # df = self.magnitude(df)
        # df = self.add_NDVI_origin_process(df)
        # df = self.add_NDVI_percentage_process(df)
        # df = self.add_SM_anomaly_process(df)
        # df = self.add_Tair_anomaly_process(df)
        # df = self.add_VPD_anomaly_process(df)
        # df = self.add_Tair_origin_process(df)

        # T.save_df(df, self.dff)
        # T.df_to_excel(df, self.dff)

        # statistic
        # self.plot_different_modes(df)
        # self.two_mode_ratio(df)
        # self.delta_hot_normal(df)
        # self.plot_delta_hot_normal()
        # self.two_region_delta_bar()
        # self.two_region_trajectory(df)
        # self.spatial_two_mode_ratio(df)
        # self.ELI_gradient(df)
        # self.ELI_gradient_ratio(df)
        # self.Drought_year_spatial_tif(df)
        # self.plot_Drought_year_spatial_tif()
        # self.pdf_Drought_year_spatial_tif()

        # self.Drought_year_NDVI_percentage_spatial_tif(df)
        # self.Drought_year_NDVI_percentage_sig_spatial_tif()
        # self.delta_percentage_sig()
        # self.plot_Drought_year_NDVI_percentage_spatial_tif()
        # self.pdf_Drought_year_spatial_tif()
        # self.bar_Drought_year_NDVI_percentage_spatial_tif()
        # self.bar_Drought_year_NDVI_percentage_spatial_tif_all_area()
        # self.AI_gradient_Drought_year_spatial_tif()
        # self.delta_area_ratio_gradient_bar()

        # self.rs_rt_area_ratio_bar(df)
        # self.delta_area_ratio_bar()
        # self.delta_area_statistic()
        # self.delta_value_statistic()
        # self.print_delta_value_statistic()
        self.check_compensation_excerbation_season()
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

    def identification_two_modes(self,df):
        mode_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            drought_year = row['drought_year']
            NDVI_progress = row['NDVI_progress'].tolist()
            NDVI_progress = np.array(NDVI_progress)
            NDVI_progress_reshape = NDVI_progress.reshape(-1, 6)
            drought_year_NDVI = NDVI_progress_reshape[1]
            max_NDVI = np.nanmax(drought_year_NDVI)
            min_NDVI = np.nanmin(drought_year_NDVI)
            max_NDVI_arg = np.nanargmax(drought_year_NDVI)
            min_NDVI_arg = np.nanargmin(drought_year_NDVI)
            if max_NDVI > 0 and min_NDVI < 0:
                if abs(max_NDVI) > abs(min_NDVI):
                    ratio = abs(max_NDVI) / abs(min_NDVI)
                else:
                    ratio = abs(min_NDVI) / abs(max_NDVI)
                # print('ratio',ratio)
                if max_NDVI_arg < min_NDVI_arg:
                    if max_NDVI_arg <= 2 and min_NDVI_arg >= 2:
                        if ratio < 3:
                            mode = 'Compensation'
                        else:
                            mode = 'other'
                    else:
                        mode = 'other'
                else:
                    mode = 'other'
            elif min_NDVI> 0:
                mode = 'Positive_effect'
            elif max_NDVI < 0:
                mode = 'Exacerbation'
            else:
                raise IOError('error')
            mode_list.append(mode)
        df['Compensation_Exacerbation'] = mode_list
        return df

    def magnitude(self,df):
        Compensation_Exacerbation_mode_list = ('Compensation', 'Exacerbation', 'Positive_effect', 'other')
        magnitude_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            mode = row['Compensation_Exacerbation']
            NDVI_progress = row['NDVI_progress'].tolist()
            NDVI_progress = np.array(NDVI_progress)
            NDVI_progress_reshape = NDVI_progress.reshape(-1, 6)
            drought_year_NDVI = NDVI_progress_reshape[1]
            max_NDVI = np.nanmax(drought_year_NDVI)
            min_NDVI = np.nanmin(drought_year_NDVI)
            if mode == 'Compensation':
                magnitude = max_NDVI - min_NDVI
            elif mode == 'Exacerbation':
                magnitude = min_NDVI
            elif mode == 'Positive_effect':
                magnitude = max_NDVI
            elif mode == 'other':
                magnitude = np.nan
            else:
                raise IOError('error')
            magnitude_list.append(magnitude)
        df['magnitude'] = magnitude_list

        return df

    def plot_different_modes(self,df):
        Compensation_Exacerbation_mode_list = ('Compensation', 'Exacerbation', 'Positive_effect', 'other')
        for mode in Compensation_Exacerbation_mode_list:
            df_mode = df[df['Compensation_Exacerbation'] == mode]
            vals_list = []
            for i,row in df_mode.iterrows():
                NDVI_progress = row['NDVI_progress'].tolist()
                NDVI_progress = np.array(NDVI_progress)
                NDVI_progress_reshape = NDVI_progress.reshape(-1, 6)
                drought_year_NDVI = NDVI_progress_reshape[1]
                vals_list.append(drought_year_NDVI)
            mean = np.nanmean(vals_list,axis=0)
            std = np.nanstd(vals_list,axis=0) / 4.
            # plt.figure()
            plt.plot(mean,label=mode)
            plt.fill_between(range(len(mean)),mean-std,mean+std,alpha=0.3)
        plt.legend()
        plt.show()

    def two_mode_ratio(self,df):
        drought_type_list = global_drought_type_list
        ELI_class_list = global_ELI_class_list
        Compensation_Exacerbation_mode_list = ('Compensation', 'Exacerbation', 'Positive_effect', 'other')

        for drt in drought_type_list:
            for ELI_class in ELI_class_list:
                df_drt = df[df['drought_type'] == drt]
                df_ELI = df_drt[df_drt['ELI_class'] == ELI_class]
                # print(drt,ELI_class,len(df_ELI))
                mode_list = df_ELI['Compensation_Exacerbation'].tolist()
                print('------------------')
                print(drt,ELI_class,len(mode_list))
                print('------------------')

                for mode in Compensation_Exacerbation_mode_list:
                    mode_count = mode_list.count(mode)
                    ratio = mode_count / len(mode_list)
                    print(mode,ratio)


    def delta_hot_normal(self,df):
        outdir = join(self.this_class_tif,'delta_hot_normal')
        T.mk_dir(outdir)
        pix_list = T.get_df_unique_val_list(df,'pix')
        # print(len(pix_list))
        drought_type_list = global_drought_type_list

        drought_year_list = range(1, 5)
        for drought_year_i in drought_year_list:
            df_group_dict = T.df_groupby(df,'pix')
            spatial_dict = {}
            for pix in tqdm(pix_list):
                df_pix = df_group_dict[pix]
                df_hot = df_pix[df_pix['drought_type'] == 'hot-drought']
                df_normal = df_pix[df_pix['drought_type'] == 'normal-drought']
                if len(df_hot) == 0 or len(df_normal) == 0:
                    continue
                # NDVI_progress = row['NDVI_progress'].tolist()
                # NDVI_progress = np.array(NDVI_progress)
                # NDVI_progress_reshape = NDVI_progress.reshape(-1, 6)
                # drought_year_NDVI = NDVI_progress_reshape[1]
                NDVI_progress_hot = df_hot['NDVI_progress'].tolist()
                NDVI_progress_normal = df_normal['NDVI_progress'].tolist()
                mean_hot = np.nanmean(NDVI_progress_hot,axis=0)
                mean_normal = np.nanmean(NDVI_progress_normal,axis=0)

                mean_hot_reshape = np.array(mean_hot).reshape(-1,6)
                mean_normal_reshape = np.array(mean_normal).reshape(-1,6)

                mean_hot_drought_year = mean_hot_reshape[1:drought_year_i+1]
                mean_normal_drought_year = mean_normal_reshape[1:drought_year_i+1]

                mean_normal_drought_NDVI = np.nanmean(mean_normal_drought_year)
                mean_hot_drought_NDVI = np.nanmean(mean_hot_drought_year)
                delta = mean_hot_drought_NDVI - mean_normal_drought_NDVI
                spatial_dict[pix] = delta
            outf = join(outdir, f'drought_year_{drought_year_i}.tif')

            DIC_and_TIF().pix_dic_to_tif(spatial_dict,outf)
            # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
            # plt.imshow(arr,vmin=-1,vmax=1,cmap='bwr_r',interpolation='nearest')
            # plt.colorbar()
            # DIC_and_TIF().plot_back_ground_arr(global_land_tif)
            # plt.show()

    def plot_delta_hot_normal(self):
        fdir = join(self.this_class_tif,'delta_hot_normal')
        outdir = join(self.this_class_png, 'delta_hot_normal')
        T.mk_dir(outdir)
        sig_tif = join(self.this_class_tif,'delta_percentage/delta_percentage_sig.tif')
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            outf = join(outdir,f.replace('.tif','.png'))
            m,ret = Plot().plot_ortho(fpath,vmin=-0.5,vmax=.5,cmap='Spectral')
            Plot().plot_ortho_significance_scatter(m,sig_tif,temp_root)
            plt.savefig(outf,dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

    def two_region_delta_bar(self):
        fdir = join(self.this_class_tif,'spatial_two_mode_magnitude')
        ELI_class_list = global_ELI_class_list
        drought_type_list = global_drought_type_list
        spatial_dicts = {}
        key_list = []
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dicts[f.replace('.tif','')] = spatial_dict
            key_list.append(f.replace('.tif',''))
        df = T.spatial_dics_to_df(spatial_dicts)
        df = Dataframe_func(df).df
        flag = 0
        for ELI_class in ELI_class_list:
            # plt.figure(figsize=(12, 6))
            df_ELI = df[df['ELI_class'] == ELI_class]
            mean_list = []
            err_list = []
            for key in key_list:
                vals = df_ELI[key].tolist()
                mean = np.nanmean(vals)
                std = np.nanstd(vals) / 8
                mean_list.append(mean)
                err_list.append(std)
            x = range(len(key_list))
            x = np.array(x)
            # plt.errorbar(x+flag*0.2,mean_list,yerr=err_list,label=ELI_class)
            plt.errorbar(x+flag*0.2,mean_list,yerr=err_list,label=ELI_class)
            plt.xticks(range(len(key_list)),key_list)
        plt.legend()
        plt.show()



    def spatial_two_mode_ratio(self,df):
        pix_list = T.get_df_unique_val_list(df,'pix')
        # print(len(pix_list))
        drought_type_list = global_drought_type_list
        df_group_dict = T.df_groupby(df,'pix')
        spatial_dict = {}
        for pix in tqdm(pix_list):
            df_pix = df_group_dict[pix]
            df_hot = df_pix[df_pix['drought_type'] == 'hot-drought']
            df_normal = df_pix[df_pix['drought_type'] == 'normal-drought']
            if len(df_hot) == 0 or len(df_normal) == 0:
                continue
            # NDVI_progress = row['NDVI_progress'].tolist()
            # NDVI_progress = np.array(NDVI_progress)
            # NDVI_progress_reshape = NDVI_progress.reshape(-1, 6)
            # drought_year_NDVI = NDVI_progress_reshape[1]
            NDVI_progress_hot = df_hot['NDVI_progress'].tolist()
            NDVI_progress_normal = df_normal['NDVI_progress'].tolist()
            mean_hot = np.nanmean(NDVI_progress_hot,axis=0)
            mean_normal = np.nanmean(NDVI_progress_normal,axis=0)

            mean_hot_reshape = np.array(mean_hot).reshape(-1,6)
            mean_normal_reshape = np.array(mean_normal).reshape(-1,6)

            mean_hot_drought_year = mean_hot_reshape[1]
            mean_normal_drought_year = mean_normal_reshape[1]

            mean_normal_drought_NDVI = np.nanmean(mean_normal_drought_year)
            Compensation_num = 0
            for i,row in df_hot.iterrows():
                NDVI_progress_hot_i = row['NDVI_progress'].tolist()
                mean_hot_reshape_i = np.array(NDVI_progress_hot_i).reshape(-1, 6)
                drought_year_NDVI_i = mean_hot_reshape_i[1]
                mean_drought_year_NDVI_i = np.nanmean(drought_year_NDVI_i)
                delta = mean_drought_year_NDVI_i - mean_normal_drought_NDVI
                if delta > 0:
                    mode = 'Compensation'
                    mode_num = 1
                elif delta < 0:
                    mode = 'Exacerbation'
                    mode_num = -1
                else:
                    mode = 'other'
                    mode_num = 0
                if mode_num == 1:
                    Compensation_num += 1
            ratio = Compensation_num / len(df_hot)
            spatial_dict[pix] = ratio
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        plt.imshow(arr,vmin=0,vmax=1,cmap='RdBu',interpolation='nearest')
        plt.colorbar()
        DIC_and_TIF().plot_back_ground_arr(global_land_tif)
        plt.show()


    def ELI_gradient(self,df):
        pix_list = T.get_df_unique_val_list(df, 'pix')
        # print(len(pix_list))
        ELI_list = global_ELI_class_list
        drought_type_list = global_drought_type_list
        df_group_dict = T.df_groupby(df, 'pix')
        spatial_dict = {}
        for pix in tqdm(pix_list):
            df_pix = df_group_dict[pix]
            df_hot = df_pix[df_pix['drought_type'] == 'hot-drought']
            df_normal = df_pix[df_pix['drought_type'] == 'normal-drought']
            if len(df_hot) == 0 or len(df_normal) == 0:
                continue
            ELI = df_pix['ELI'].tolist()[0]
            aridity_index = df_pix['aridity_index'].tolist()[0]
            # NDVI_progress = row['NDVI_progress'].tolist()
            # NDVI_progress = np.array(NDVI_progress)
            # NDVI_progress_reshape = NDVI_progress.reshape(-1, 6)
            # drought_year_NDVI = NDVI_progress_reshape[1]
            NDVI_progress_hot = df_hot['NDVI_progress'].tolist()
            NDVI_progress_normal = df_normal['NDVI_progress'].tolist()
            mean_hot = np.nanmean(NDVI_progress_hot, axis=0)
            mean_normal = np.nanmean(NDVI_progress_normal, axis=0)

            mean_hot_reshape = np.array(mean_hot).reshape(-1, 6)
            mean_normal_reshape = np.array(mean_normal).reshape(-1, 6)

            mean_hot_drought_year = mean_hot_reshape[1]
            mean_normal_drought_year = mean_normal_reshape[1]

            mean_normal_drought_NDVI = np.nanmean(mean_normal_drought_year)
            mean_hot_drought_NDVI = np.nanmean(mean_hot_drought_year)
            delta = mean_hot_drought_NDVI - mean_normal_drought_NDVI
            # result_dict = {'ELI': ELI, 'delta': delta}
            result_dict = {'ELI': aridity_index, 'delta': delta}
            spatial_dict[pix] = result_dict
        df = T.dic_to_df(spatial_dict,'pix')
        df = df[df['ELI']<3]
        x = df['ELI'].tolist()
        y = df['delta'].tolist()
        # KDE_plot().plot_scatter_hex(x,y,xlim=(-1,1),ylim=(-1,1))
        a, b, r, p = KDE_plot().plot_scatter(x,y,plot_fit_line=True,s=1)
        print(a,b,r,p)
        plt.show()

    def ELI_gradient_ratio(self,df):
        outdir = join(self.this_class_png,'ELI_gradient_ratio')
        T.mk_dir(outdir)
        pix_list = T.get_df_unique_val_list(df, 'pix')
        # print(len(pix_list))
        ELI_list = global_ELI_class_list
        drought_type_list = global_drought_type_list
        df_group_dict = T.df_groupby(df, 'pix')
        drought_year = range(1,5)
        for drought_year_i in drought_year:
            spatial_dict = {}
            for pix in tqdm(pix_list):
                df_pix = df_group_dict[pix]
                df_hot = df_pix[df_pix['drought_type'] == 'hot-drought']
                df_normal = df_pix[df_pix['drought_type'] == 'normal-drought']
                if len(df_hot) == 0 or len(df_normal) == 0:
                    continue
                ELI = df_pix['ELI'].tolist()[0]
                aridity_index = df_pix['aridity_index'].tolist()[0]
                # NDVI_progress = row['NDVI_progress'].tolist()
                # NDVI_progress = np.array(NDVI_progress)
                # NDVI_progress_reshape = NDVI_progress.reshape(-1, 6)
                # drought_year_NDVI = NDVI_progress_reshape[1]
                NDVI_progress_hot = df_hot['NDVI_progress'].tolist()
                NDVI_progress_normal = df_normal['NDVI_progress'].tolist()
                mean_hot = np.nanmean(NDVI_progress_hot, axis=0)
                mean_normal = np.nanmean(NDVI_progress_normal, axis=0)

                mean_hot_reshape = np.array(mean_hot).reshape(-1, 6)
                mean_normal_reshape = np.array(mean_normal).reshape(-1, 6)

                mean_hot_drought_year = mean_hot_reshape[1:drought_year_i+1]
                mean_normal_drought_year = mean_normal_reshape[1:drought_year_i+1]

                mean_normal_drought_NDVI = np.nanmean(mean_normal_drought_year)
                mean_hot_drought_NDVI = np.nanmean(mean_hot_drought_year)
                delta = mean_hot_drought_NDVI - mean_normal_drought_NDVI
                # result_dict = {'ELI': ELI, 'delta': delta}
                result_dict = {'ELI': aridity_index, 'delta': delta}
                spatial_dict[pix] = result_dict
            df = T.dic_to_df(spatial_dict,'pix')
            df = df[df['ELI']<3]
            ELI_bins = np.arange(0.1,3,0.2)
            df_group,bins_list_str = T.df_bin(df,'ELI',ELI_bins)
            x = []
            y = []
            for name,df_group_i in df_group:
                x_i = name[0].left
                vals = df_group_i['delta'].tolist()
                # mean = np.nanmean(vals)
                vals = np.array(vals)
                positive_vals_count = len(vals[vals>0])
                ratio = positive_vals_count/len(vals)
                x.append(x_i)
                y.append(ratio)
            y_smooth = SMOOTH().smooth_convolve(y,window_len=5)
            # y_smooth = y
            plt.plot(x,y_smooth,label=drought_year_i)
        plt.legend()
        outf = join(outdir,'ELI_gradient_ratio_5.pdf')
        plt.savefig(outf)
        plt.close()
        T.open_path_and_file(outdir)
        # plt.show()

        # x = df['ELI'].tolist()
        # y = df['delta'].tolist()
        # # KDE_plot().plot_scatter_hex(x,y,xlim=(-1,1),ylim=(-1,1))
        # a, b, r, p = KDE_plot().plot_scatter(x,y,plot_fit_line=True,s=1)
        # print(a,b,r,p)
        # plt.show()

    def two_region_trajectory(self,df):
        outdir = join(self.this_class_png,'two_region_trajectory')
        T.mk_dir(outdir)
        ELI_class_list = global_ELI_class_list
        drought_type_list = global_drought_type_list

        for ELI in ELI_class_list:
            plt.figure(figsize=(9*centimeter_factor,6*centimeter_factor))
            for drt in drought_type_list:
                df_ELI = df[df['ELI_class']==ELI]
                df_ELI_drt = df_ELI[df_ELI['drought_type']==drt]
                NDVI = df_ELI_drt['NDVI_progress'].tolist()
                NDVI = np.array(NDVI)
                mean = np.nanmean(NDVI,axis=0)
                std = np.nanstd(NDVI,axis=0) / 8
                plt.plot(mean,label=drt)
                plt.fill_between(np.arange(len(mean)),mean-std,mean+std,alpha=0.3)
                plt.xticks(np.arange(len(mean))[::6],np.arange(-1,5))
            outf = join(outdir,'{}.pdf'.format(ELI))
            # plt.legend()
            plt.savefig(outf)
            plt.close()
        T.open_path_and_file(outdir)

        pass

    def Drought_year_spatial_tif(self,df):
        outdir = join(self.this_class_tif,'Drought_year_spatial_tif')
        T.mk_dir(outdir)
        drought_type_list = global_drought_type_list
        for drt in drought_type_list:
            df_drt = df[df['drought_type']==drt]
            df_pix = T.df_groupby(df_drt,'pix')
            spatial_dict = {}
            for pix in tqdm(df_pix,desc=drt):
                df_pix_i = df_pix[pix]
                NDVI_progress = df_pix_i['NDVI_progress'].tolist()
                mean_progress = np.nanmean(NDVI_progress, axis=0)
                mean_progress_reshape = np.array(mean_progress).reshape(-1, 6)
                mean_progress_reshape_drought_year = mean_progress_reshape[1]
                mean_drought_year_NDVI = np.nanmean(mean_progress_reshape_drought_year)
                spatial_dict[pix] = mean_drought_year_NDVI
            outf = join(outdir,'{}.tif'.format(drt))
            DIC_and_TIF().pix_dic_to_tif(spatial_dict,outf)


    def plot_Drought_year_spatial_tif(self):
        fdir = join(self.this_class_tif,'Drought_year_spatial_tif')
        fdir_sig = join(self.this_class_tif,'Drought_year_NDVI_percentage_sig_spatial_tif')

        outdir = join(self.this_class_png,'Drought_year_spatial_tif')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            fpath_sig = join(fdir_sig,f)
            if not f.endswith('.tif'):
                continue
            outpath = join(outdir,f'{f.replace(".tif","")}.png')
            m, ret = Plot().plot_ortho(fpath,vmin=-1,vmax=1,cmap='RdBu')
            Plot().plot_ortho_significance_scatter(m, fpath_sig, temp_root)

            plt.savefig(outpath,dpi=300)
            # plt.show()
            plt.close()
        T.open_path_and_file(outdir)

    def plot_Drought_year_spatial_tif_sig(self):
        fdir = join(self.this_class_tif,'Drought_year_spatial_tif')
        outdir = join(self.this_class_png,'plot_Drought_year_spatial_tif_sig')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            if not f.endswith('.tif'):
                continue
            outpath = join(outdir,f'{f.replace(".tif","")}.png')
            Plot().plot_ortho(fpath,vmin=-1,vmax=1,cmap='RdBu')
            plt.savefig(outpath,dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

    def pdf_Drought_year_spatial_tif(self):
        delta_fpath = join(self.this_class_tif,'spatial_two_mode_magnitude/spatial_two_mode_magnitude_1.tif')
        fdir = join(self.this_class_tif,'Drought_year_spatial_tif')
        outdir = join(self.this_class_png,'pdf_Drought_year_spatial_tif')
        T.mk_dir(outdir)
        spatial_dics = {}
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            if not f.endswith('.tif'):
                continue
            key = f.replace('.tif','')
            spatial_dic = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dics[key] = spatial_dic
        spatial_dics['delta'] = DIC_and_TIF().spatial_tif_to_dic(delta_fpath)
        df = T.spatial_dics_to_df(spatial_dics)
        df = Dataframe_func(df).df
        print('-----')
        ELI_class_list = global_ELI_class_list

        for key in spatial_dics:
            plt.figure(figsize=(9*centimeter_factor,6*centimeter_factor))
            for ELI in ELI_class_list:
                df_ELI = df[df['ELI_class']==ELI]
                df_ELI = df_ELI[df_ELI[key]>-2]
                df_ELI = df_ELI[df_ELI[key]<2]
                vals = df_ELI[key].tolist()
                x1, y1 = Plot().plot_hist_smooth(vals,bins=100,alpha=0)
                plt.plot(x1,y1,label=ELI)
            plt.legend()
            plt.title(key)
            outf = join(outdir,'{}.pdf'.format(key))
            # plt.savefig(outf)
            # plt.close()
            plt.show()

        # T.open_path_and_file(outdir)
    def add_NDVI_origin_process(self,df):
        # df = Load_dataframe()
        NDVI_spatial_dict,_ = Load_Data().NDVI_origin()
        year_list = global_year_range_list
        gs = global_gs
        NDVI_list_all = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            NDVI = NDVI_spatial_dict[pix]
            NDVI = np.array(NDVI,dtype=float)
            NDVI[NDVI>10000] = np.nan
            NDVI[NDVI<0] = np.nan
            NDVI_gs = T.monthly_vals_to_annual_val(NDVI,gs,method='array')
            NDVI_gs_dict = T.dict_zip(year_list,NDVI_gs)
            NDVI_list = []
            year_list_i = []
            for y in range(-1,5):
                y_i = drought_year+y
                if y_i in NDVI_gs_dict:
                    NDVI_list.append(NDVI_gs_dict[drought_year+y])
                else:
                    NDVI_list.append([np.nan]*len(gs))
                year_list_i.append(y_i)
            NDVI_list = np.array(NDVI_list)
            NDVI_list = NDVI_list.flatten()
            NDVI_list_all.append(NDVI_list)
        df['NDVI_origin_progress'] = NDVI_list_all
        return df

    def add_SM_anomaly_process(self,df):
        # df = Load_dataframe()
        NDVI_spatial_dict,data_name = Load_Data().GLEAM_SMRoot_anomaly_detrend()
        # print(data_name)
        # exit()
        year_list = global_year_range_list
        gs = global_gs
        NDVI_list_all = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            NDVI = NDVI_spatial_dict[pix]
            NDVI = np.array(NDVI,dtype=float)
            NDVI[NDVI>10000] = np.nan
            # NDVI[NDVI<0] = np.nan
            NDVI_gs = T.monthly_vals_to_annual_val(NDVI,gs,method='array')
            NDVI_gs_dict = T.dict_zip(year_list,NDVI_gs)
            NDVI_list = []
            year_list_i = []
            for y in range(-1,5):
                y_i = drought_year+y
                if y_i in NDVI_gs_dict:
                    NDVI_list.append(NDVI_gs_dict[drought_year+y])
                else:
                    NDVI_list.append([np.nan]*len(gs))
                year_list_i.append(y_i)
            NDVI_list = np.array(NDVI_list)
            NDVI_list = NDVI_list.flatten()
            NDVI_list_all.append(NDVI_list)
        df[f'{data_name}_progress'] = NDVI_list_all
        return df


    def add_Tair_anomaly_process(self,df):
        # df = Load_dataframe()
        NDVI_spatial_dict,data_name = Load_Data().Temperature_anomaly_detrend()
        # print(data_name)
        # exit()
        year_list = global_year_range_list
        gs = global_gs
        NDVI_list_all = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            NDVI = NDVI_spatial_dict[pix]
            NDVI = np.array(NDVI,dtype=float)
            NDVI[NDVI>10000] = np.nan
            # NDVI[NDVI<0] = np.nan
            NDVI_gs = T.monthly_vals_to_annual_val(NDVI,gs,method='array')
            NDVI_gs_dict = T.dict_zip(year_list,NDVI_gs)
            NDVI_list = []
            year_list_i = []
            for y in range(-1,5):
                y_i = drought_year+y
                if y_i in NDVI_gs_dict:
                    NDVI_list.append(NDVI_gs_dict[drought_year+y])
                else:
                    NDVI_list.append([np.nan]*len(gs))
                year_list_i.append(y_i)
            NDVI_list = np.array(NDVI_list)
            NDVI_list = NDVI_list.flatten()
            NDVI_list_all.append(NDVI_list)
        df[f'{data_name}_progress'] = NDVI_list_all
        return df

    def add_Tair_origin_process(self,df):
        # df = Load_dataframe()
        NDVI_spatial_dict,data_name = Load_Data().Temperature_origin()
        # print(data_name)
        # exit()
        year_list = global_year_range_list
        gs = global_gs
        NDVI_list_all = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            NDVI = NDVI_spatial_dict[pix]
            NDVI = np.array(NDVI,dtype=float)
            NDVI[NDVI>10000] = np.nan
            # NDVI[NDVI<0] = np.nan
            NDVI_gs = T.monthly_vals_to_annual_val(NDVI,gs,method='array')
            NDVI_gs_dict = T.dict_zip(year_list,NDVI_gs)
            NDVI_list = []
            year_list_i = []
            for y in range(-1,5):
                y_i = drought_year+y
                if y_i in NDVI_gs_dict:
                    NDVI_list.append(NDVI_gs_dict[drought_year+y])
                else:
                    NDVI_list.append([np.nan]*len(gs))
                year_list_i.append(y_i)
            NDVI_list = np.array(NDVI_list)
            NDVI_list = NDVI_list.flatten()
            NDVI_list_all.append(NDVI_list)
        df[f'{data_name}_progress'] = NDVI_list_all
        return df

    def add_VPD_anomaly_process(self,df):
        # df = Load_dataframe()
        NDVI_spatial_dict,data_name = Load_Data().VPD_anomaly_detrend()
        # print(data_name)
        # exit()
        year_list = global_year_range_list
        gs = global_gs
        NDVI_list_all = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            NDVI = NDVI_spatial_dict[pix]
            NDVI = np.array(NDVI,dtype=float)
            NDVI[NDVI>10000] = np.nan
            # NDVI[NDVI<0] = np.nan
            NDVI_gs = T.monthly_vals_to_annual_val(NDVI,gs,method='array')
            NDVI_gs_dict = T.dict_zip(year_list,NDVI_gs)
            NDVI_list = []
            year_list_i = []
            for y in range(-1,5):
                y_i = drought_year+y
                if y_i in NDVI_gs_dict:
                    NDVI_list.append(NDVI_gs_dict[drought_year+y])
                else:
                    NDVI_list.append([np.nan]*len(gs))
                year_list_i.append(y_i)
            NDVI_list = np.array(NDVI_list)
            NDVI_list = NDVI_list.flatten()
            NDVI_list_all.append(NDVI_list)
        df[f'{data_name}_progress'] = NDVI_list_all
        return df

    def climatology_percentage(self, vals):
        '''
        percentage
        :param vals:
        :return:
        '''
        pix_percentage = []
        climatology_means = []
        for m in range(1, 13):
            one_mon = []
            for i in range(len(vals)):
                mon = i % 12 + 1
                if mon == m:
                    one_mon.append(vals[i])
            mean = np.nanmean(one_mon)
            climatology_means.append(mean)
        for i in range(len(vals)):
            mon = i % 12
            mean_ = climatology_means[mon]
            percentage = vals[i] / mean_ * 100 - 100
            pix_percentage.append(percentage)
        pix_percentage = np.array(pix_percentage)
        return pix_percentage

    def add_NDVI_percentage_process(self,df):
        # df = Load_dataframe()
        NDVI_spatial_dict,_ = Load_Data().NDVI_origin()
        NDVI_percentage_spatial_dict = {}
        for pix in tqdm(NDVI_spatial_dict):
            NDVI = NDVI_spatial_dict[pix]
            NDVI[NDVI > 10000] = np.nan
            NDVI[NDVI < 0] = np.nan
            if T.is_all_nan(NDVI):
                continue
            percentage = self.climatology_percentage(NDVI)
            NDVI_percentage_spatial_dict[pix] = percentage

        year_list = global_year_range_list
        gs = global_gs
        NDVI_list_all = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            NDVI_percentage = NDVI_percentage_spatial_dict[pix]
            NDVI_percentage = np.array(NDVI_percentage,dtype=float)
            # plt.plot(NDVI_percentage)
            # plt.show()
            NDVI_gs = T.monthly_vals_to_annual_val(NDVI_percentage,gs,method='array')
            NDVI_gs_dict = T.dict_zip(year_list,NDVI_gs)
            NDVI_list = []
            year_list_i = []
            for y in range(-1,5):
                y_i = drought_year+y
                if y_i in NDVI_gs_dict:
                    NDVI_list.append(NDVI_gs_dict[drought_year+y])
                else:
                    NDVI_list.append([np.nan]*len(gs))
                year_list_i.append(y_i)
            NDVI_list = np.array(NDVI_list)
            NDVI_list = NDVI_list.flatten()
            NDVI_list_all.append(NDVI_list)
        df['NDVI_percentage_progress'] = NDVI_list_all
        return df

    def Drought_year_NDVI_percentage_spatial_tif(self,df):
        outdir = join(self.this_class_tif,'Drought_year_NDVI_percentage_spatial_tif')
        NDVI_data_dict = Load_Data().NDVI_origin()
        T.mk_dir(outdir)
        drought_type_list = global_drought_type_list
        for drt in drought_type_list:
            df_drt = df[df['drought_type']==drt]
            df_pix = T.df_groupby(df_drt,'pix')
            spatial_dict = {}
            for pix in tqdm(df_pix,desc=drt):
                df_pix_i = df_pix[pix]
                NDVI_progress = df_pix_i['NDVI_percentage_progress'].tolist()
                # plt.plot(NDVI_progress)
                # plt.show()
                mean_progress = np.nanmean(NDVI_progress, axis=0)
                mean_progress_reshape = np.array(mean_progress).reshape(-1, 6)
                mean_progress_reshape_drought_year = mean_progress_reshape[1]
                mean_drought_year_NDVI = np.nanmean(mean_progress_reshape_drought_year)
                if mean_drought_year_NDVI > 50:
                    continue
                if mean_drought_year_NDVI < -50:
                    continue
                spatial_dict[pix] = mean_drought_year_NDVI
            outf = join(outdir,'{}.tif'.format(drt))
            DIC_and_TIF().pix_dic_to_tif(spatial_dict,outf)
        T.open_path_and_file(outdir)

    def Drought_year_NDVI_percentage_sig_spatial_tif(self):
        fdir = join(self.this_class_tif,'Drought_year_NDVI_percentage_spatial_tif')
        outdir = join(self.this_class_tif,'Drought_year_NDVI_percentage_sig_spatial_tif')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            df = T.spatial_dics_to_df({'NDVI':spatial_dict})

            sig_list = []
            for i,row in df.iterrows():
                val = row['NDVI']
                if val > 5 or val < -5:
                    sig_list.append(0.0001)
                else:
                    sig_list.append(1)
            df['sig'] = sig_list
            spatial_dict_sig = T.df_to_spatial_dic(df,'sig')
            # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_sig)
            DIC_and_TIF().pix_dic_to_tif(spatial_dict_sig,outpath)

    def delta_percentage_sig(self):
        fdir = join(self.this_class_tif,'Drought_year_NDVI_percentage_spatial_tif')
        outdir = join(self.this_class_tif,'delta_percentage')
        T.mk_dir(outdir)
        fpath_hot = join(fdir,'hot-drought.tif')
        fpath_normal = join(fdir,'normal-drought.tif')
        spatial_dic_hot = DIC_and_TIF().spatial_tif_to_dic(fpath_hot)
        spatial_dic_normal = DIC_and_TIF().spatial_tif_to_dic(fpath_normal)
        spatial_dic_delta = {}
        for pix in spatial_dic_hot:
            delta = spatial_dic_hot[pix] - spatial_dic_normal[pix]
            if delta >5 or delta < -5:
                sig_val = 0.0001
            else:
                sig_val = 1
            spatial_dic_delta[pix] = sig_val
        outf = join(outdir,'delta_percentage_sig.tif')
        DIC_and_TIF().pix_dic_to_tif(spatial_dic_delta,outf)
        pass

    def plot_Drought_year_NDVI_percentage_spatial_tif(self):
        fdir = join(self.this_class_tif,'Drought_year_NDVI_percentage_spatial_tif')
        fdir_sig = join(self.this_class_tif,'Drought_year_NDVI_percentage_sig_spatial_tif')
        outdir = join(self.this_class_png,'Drought_year_NDVI_percentage_spatial_tif')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            fpath_sig = join(fdir_sig,f)
            if not f.endswith('.tif'):
                continue
            outpath = join(outdir,f'{f.replace(".tif","")}.png')
            m, ret = Plot().plot_ortho(fpath,vmin=-20,vmax=20,cmap='RdBu')
            Plot().plot_ortho_significance_scatter(m,fpath_sig,temp_root)
            # plt.show()
            plt.savefig(outpath,dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

    def pdf_Drought_year_spatial_tif1(self):
        delta_fpath = join(self.this_class_tif,'spatial_two_mode_magnitude/spatial_two_mode_magnitude_1.tif')
        fdir = join(self.this_class_tif,'Drought_year_spatial_tif')
        outdir = join(self.this_class_png,'pdf_Drought_year_spatial_tif')
        T.mk_dir(outdir)
        spatial_dics = {}
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            if not f.endswith('.tif'):
                continue
            key = f.replace('.tif','')
            spatial_dic = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dics[key] = spatial_dic
        spatial_dics['delta'] = DIC_and_TIF().spatial_tif_to_dic(delta_fpath)
        df = T.spatial_dics_to_df(spatial_dics)
        df = Dataframe_func(df).df
        print('-----')
        ELI_class_list = global_ELI_class_list

        for key in spatial_dics:
            plt.figure(figsize=(9*centimeter_factor,6*centimeter_factor))
            for ELI in ELI_class_list:
                df_ELI = df[df['ELI_class']==ELI]
                df_ELI = df_ELI[df_ELI[key]>-2]
                df_ELI = df_ELI[df_ELI[key]<2]
                vals = df_ELI[key].tolist()
                x1, y1 = Plot().plot_hist_smooth(vals,bins=100,alpha=0)
                plt.plot(x1,y1,label=ELI)
            plt.legend()
            plt.title(key)
            outf = join(outdir,'{}.pdf'.format(key))
            # plt.savefig(outf)
            # plt.close()
            plt.show()

        # T.open_path_and_file(outdir)

    def bar_Drought_year_NDVI_percentage_spatial_tif(self):
        fdir = join(self.this_class_tif,'Drought_year_NDVI_percentage_spatial_tif')
        outdir = join(self.this_class_png,'Drought_year_NDVI_percentage_spatial_tif')
        T.mk_dir(outdir)
        spatial_dics = {}
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            if not f.endswith('.tif'):
                continue
            key = f.replace('.tif','')
            spatial_dic = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dics[key] = spatial_dic
        df = T.spatial_dics_to_df(spatial_dics)
        df = Dataframe_func(df).df
        print('-----')
        T.print_head_n(df)
        # exit()
        ELI_class_list = global_ELI_class_list
        plt.figure(figsize=(15 * centimeter_factor, 6 * centimeter_factor))
        for key in spatial_dics:
            for ELI in ELI_class_list:
                df_ELI = df[df['ELI_class']==ELI]
                df_ELI = df_ELI[df_ELI[key]>-30]
                df_ELI = df_ELI[df_ELI[key]<30]
                vals = df_ELI[key].tolist()
                vals = np.array(vals)
                strong_positive_vals_count = len(vals[vals>5])
                positive_vals_count = len(vals[vals>0])
                negative_vals_count = len(vals[vals<0])
                strong_negative_vals_count = len(vals[vals<-5])
                total_count = len(vals)

                strong_pos_ratio = strong_positive_vals_count / total_count * 100
                pos_ratio = positive_vals_count / total_count * 100
                neg_ratio = negative_vals_count / total_count * 100
                strong_neg_ratio = strong_negative_vals_count / total_count * 100

                plt.barh(f'{ELI}_{key}',strong_pos_ratio,color='g',zorder=1,linewidth=1,edgecolor='k',height=.5)
                plt.barh(f'{ELI}_{key}',pos_ratio,color='g',zorder=0,linewidth=1,edgecolor='k',height=.5,alpha=0.05)
                plt.barh(f'{ELI}_{key}',pos_ratio,color='none',zorder=2,linewidth=1,edgecolor='k',height=.5)
                plt.barh(f'{ELI}_{key}',-neg_ratio,color='r',zorder=0,linewidth=1,edgecolor='k',height=.5,alpha=0.05)
                plt.barh(f'{ELI}_{key}',-neg_ratio,color='none',zorder=2,linewidth=1,edgecolor='k',height=.5)
                plt.barh(f'{ELI}_{key}',-strong_neg_ratio,color='r',zorder=1,linewidth=1,edgecolor='k',height=.5)
                plt.text(pos_ratio,f'{ELI}_{key}',f'{pos_ratio:.0f} ({strong_pos_ratio:.0f})',ha='left',va='center')
                plt.text(-neg_ratio,f'{ELI}_{key}',f'{neg_ratio:.0f} ({strong_neg_ratio:.0f})',ha='right',va='center')
        plt.xlim(-200,200)
        plt.tight_layout()
        # plt.show()
        outf = join(outdir,'bar_Drought_year_NDVI_percentage_spatial_tif.pdf')
        plt.savefig(outf)
        T.open_path_and_file(outdir)

    def bar_Drought_year_NDVI_percentage_spatial_tif_all_area(self):
        fdir = join(self.this_class_tif,'Drought_year_NDVI_percentage_spatial_tif')
        outdir = join(self.this_class_png,'bar_Drought_year_NDVI_percentage_spatial_tif_all_area')
        T.mk_dir(outdir)
        spatial_dics = {}
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            if not f.endswith('.tif'):
                continue
            key = f.replace('.tif','')
            spatial_dic = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dics[key] = spatial_dic
        df = T.spatial_dics_to_df(spatial_dics)
        df = Dataframe_func(df).df
        print('-----')
        T.print_head_n(df)
        # exit()
        plt.figure(figsize=(15 * centimeter_factor, 6 * centimeter_factor))
        ELI = 'all-area'
        for key in spatial_dics:
            df_ELI = df
            df_ELI = df_ELI[df_ELI[key]>-30]
            df_ELI = df_ELI[df_ELI[key]<30]
            vals = df_ELI[key].tolist()
            vals = np.array(vals)
            strong_positive_vals_count = len(vals[vals>5])
            positive_vals_count = len(vals[vals>0])
            negative_vals_count = len(vals[vals<0])
            strong_negative_vals_count = len(vals[vals<-5])
            total_count = len(vals)

            strong_pos_ratio = strong_positive_vals_count / total_count * 100
            pos_ratio = positive_vals_count / total_count * 100
            neg_ratio = negative_vals_count / total_count * 100
            strong_neg_ratio = strong_negative_vals_count / total_count * 100

            plt.barh(f'{ELI}_{key}',strong_pos_ratio,color='g',zorder=1,linewidth=1,edgecolor='k',height=.5)
            plt.barh(f'{ELI}_{key}',pos_ratio,color='g',zorder=0,linewidth=1,edgecolor='k',height=.5,alpha=0.05)
            plt.barh(f'{ELI}_{key}',pos_ratio,color='none',zorder=2,linewidth=1,edgecolor='k',height=.5)
            plt.barh(f'{ELI}_{key}',-neg_ratio,color='r',zorder=0,linewidth=1,edgecolor='k',height=.5,alpha=0.05)
            plt.barh(f'{ELI}_{key}',-neg_ratio,color='none',zorder=2,linewidth=1,edgecolor='k',height=.5)
            plt.barh(f'{ELI}_{key}',-strong_neg_ratio,color='r',zorder=1,linewidth=1,edgecolor='k',height=.5)
            plt.text(pos_ratio,f'{ELI}_{key}',f'{pos_ratio:.0f} ({strong_pos_ratio:.0f})',ha='left',va='center')
            plt.text(-neg_ratio,f'{ELI}_{key}',f'{neg_ratio:.0f} ({strong_neg_ratio:.0f})',ha='right',va='center')
        plt.xlim(-200,200)
        plt.tight_layout()
        # plt.show()
        outf = join(outdir,'bar_Drought_year_NDVI_percentage_spatial_tif.pdf')
        plt.savefig(outf)
        T.open_path_and_file(outdir)

    def AI_gradient_Drought_year_spatial_tif(self,delta_fpath):
        spatial_dics = {}
        spatial_dics['delta'] = DIC_and_TIF().spatial_tif_to_dic(delta_fpath)
        df = T.spatial_dics_to_df(spatial_dics)
        df = Dataframe_func(df).df
        print('-----')
        # ELI_class_list = global_ELI_class_list
        key = 'delta'
        # plt.figure(figsize=(9*centimeter_factor,6*centimeter_factor))
        AI_bins = np.arange(0.1, 3, 0.1)
        df_group,bins_list_str = T.df_bin(df, 'aridity_index', AI_bins)
        x_list = []
        y_list = []
        error_list = []
        for name,df_group_i in df_group:
            vals = df_group_i[key].tolist()
            mean = np.nanmean(vals)
            error = np.nanstd(vals) / 4
            x = name[0].left
            x_list.append(x)
            y_list.append(mean)
            error_list.append(error)
        # plt.plot(x_list,y_list)
        plt.errorbar(x_list,y_list,yerr=error_list,capsize=2)
        plt.ylim(-0.55,0.55)
        # plt.show()
        # outf = join(outdir,'{}.pdf'.format(key))
        # plt.savefig(outf)
        # plt.close()
        # T.open_path_and_file(outdir)


    def rs_rt_area_ratio_bar(self,df):
        outdir = join(self.this_class_png, 'rs_rt_area_ratio_bar')
        T.mk_dir(outdir)
        df = Load_dataframe()
        threshold_list = [-np.inf,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,np.inf]
        # threshold_list = [-np.inf,-0.2,-0.1,-0.05]
        threshold_list = np.array(threshold_list)
        threshold_list = threshold_list + 1
        drought_type_list = global_drought_type_list
        ELI_class_list = global_ELI_class_list
        bottom = 0
        for i in range(len(threshold_list)):
            if i + 1 == len(threshold_list):
                break
            x_list = []
            y_list = []
            for ELI_class in ELI_class_list:
                for drt in drought_type_list:
                    df_ELI = df[df['ELI_class'] == ELI_class]
                    df_drt = df_ELI[df_ELI['drought_type'] == drt]
                    df_drt_copy = df_drt.copy()
                    df_drt_copy = df_drt_copy.dropna(subset=['ELI', 'rt'], how='any')
                    vals = df_drt_copy['rt']
                    vals = np.array(vals)
                    threshold_left = threshold_list[i]
                    threshold_right = threshold_list[i+1]

                    vals = vals[vals < threshold_right]
                    vals = vals[vals > threshold_left]
                    # vals = vals[vals < (1 - threshold)]
                    # vals = vals[vals > (1 + threshold)]
                    ratio = len(vals) / len(df_drt_copy) * 100
                    x_list.append(f'{ELI_class}_{drt}')
                    y_list.append(ratio)
            y_list = np.array(y_list)
            # y_list = y_list + bottom
            plt.bar(x_list,y_list,bottom=bottom,label=threshold_list[i+1])
            bottom = bottom + y_list
        plt.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()
        # outf = join(outdir,'rs_rt_area_ratio_bar.pdf')
        # plt.savefig(outf)
        # plt.close()
        # T.open_path_and_file(outdir)
        # plt.show()

    def delta_area_ratio_bar(self):
        outdir = join(self.this_class_png, 'delta_area_ratio_bar')
        T.mk_dir(outdir)
        tif = join(self.this_class_tif,'spatial_two_mode_magnitude/spatial_two_mode_magnitude_1.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(tif)
        df = T.spatial_dics_to_df({'delta':spatial_dict})
        df = Dataframe_func(df).df
        # threshold_list = [-np.inf,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,np.inf]
        # threshold_list = [-np.inf,-1,-0.8,-0.6,-0.4,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.4,0.6,0.8,1,np.inf]
        threshold_list = [-np.inf,-0.5,-0.25,0,0.25,0.5,np.inf]
        color_list = T.gen_colors(len(threshold_list))
        # threshold_list = [-np.inf,-0.2,-0.1,-0.05]
        # threshold_list = np.array(threshold_list)
        # threshold_list = threshold_list + 1
        ELI_class_list = global_ELI_class_list[::-1]
        bottom = 0
        y_list_all = []
        for i in range(len(threshold_list)):
            if i + 1 == len(threshold_list):
                break
            x_list = []
            y_list = []

            for ELI_class in ELI_class_list:
                df_ELI = df[df['ELI_class'] == ELI_class]
                df_ELI_copy = df_ELI.copy()
                df_ELI_copy = df_ELI_copy.dropna(subset=['ELI', 'delta'], how='any')
                vals = df_ELI_copy['delta']
                vals = np.array(vals)
                threshold_left = threshold_list[i]
                threshold_right = threshold_list[i+1]

                vals = vals[vals < threshold_right]
                vals = vals[vals > threshold_left]
                # vals = vals[vals < (1 - threshold)]
                # vals = vals[vals > (1 + threshold)]
                ratio = len(vals) / len(df_ELI_copy) * 100

                x_list.append(f'{ELI_class}')
                y_list.append(ratio)

            y_list = np.array(y_list)
            y_list_all.append(y_list)
            # y_list = y_list + bottom
            plt.bar(x_list,y_list,bottom=bottom,label=f'{threshold_list[i]}~{threshold_list[i+1]}',color=color_list[i],
                    width=0.3)
            bottom = bottom + y_list

        plt.legend()
        plt.xticks(rotation=0)
        plt.xlim(-1,2)
        plt.tight_layout()
        y_position1 = 0
        y_position2 = 0
        for yi in range(len(y_list_all)):

            plt.text(0, y_position1, f'{y_list_all[yi][0]:.2f}', ha='center', va='bottom', fontsize=8)
            plt.text(1, y_position2, f'{y_list_all[yi][1]:.2f}', ha='center', va='bottom', fontsize=8)
            y_position1 = y_position1 + y_list_all[yi][0]
            y_position2 = y_position2 + y_list_all[yi][1]

        outf = join(outdir,'delta_area_ratio_bar.pdf')
        # plt.savefig(outf)
        # plt.close()
        # T.open_path_and_file(outdir)
        plt.show()

    def delta_area_ratio_gradient_bar(self):
        outdir = join(self.this_class_png, 'delta_area_ratio_gradient_bar')
        T.mk_dir(outdir)
        delta_tif = join(self.this_class_tif,'delta_hot_normal','drought_year_1.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(delta_tif)
        df = T.spatial_dics_to_df({'delta':spatial_dict})
        df = Dataframe_func(df).df
        # threshold_list = [-np.inf,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,np.inf]
        # threshold_list = [-np.inf,-1,-0.8,-0.6,-0.4,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.4,0.6,0.8,1,np.inf]
        threshold_list = [-np.inf,-0.5,-0.25,0,0.25,0.5,np.inf]
        color_list = T.gen_colors(len(threshold_list)-1)
        # threshold_list = [-np.inf,-0.2,-0.1,-0.05]
        # threshold_list = np.array(threshold_list)
        # threshold_list = threshold_list + 1
        plt.figure(figsize=(15*centimeter_factor,6*centimeter_factor))
        bottom = 0
        y_list_all = []
        for i in range(len(threshold_list)):
            if i + 1 == len(threshold_list):
                break
            x_list = []
            y_list = []
            AI_bins = np.arange(0.1, 3, 0.1)
            df_group, bins_list_str = T.df_bin(df, 'aridity_index', AI_bins)
            for name, df_group_i in df_group:
                x = name[0].left
                df_group_i = df_group_i.dropna(subset=['delta'], how='any')
                vals = df_group_i['delta']
                vals = np.array(vals)
                threshold_left = threshold_list[i]
                threshold_right = threshold_list[i+1]

                vals = vals[vals < threshold_right]
                vals = vals[vals > threshold_left]
                # vals = vals[vals < (1 - threshold)]
                # vals = vals[vals > (1 + threshold)]
                ratio = len(vals) / len(df_group_i) * 100

                x_list.append(x)
                y_list.append(ratio)

            y_list = np.array(y_list)
            y_list_all.append(y_list)
            # y_list = y_list + bottom
            plt.bar(x_list,y_list,bottom=bottom,label=f'{threshold_list[i]}~{threshold_list[i+1]}',color=color_list[i],width=0.08)
                    # width=0.1,edgecolor='k')
            bottom = bottom + y_list
            print(bottom)



        # plt.legend()
        plt.xticks(rotation=0)
        plt.ylim(0, 100)
        # plt.xlim(-1,2)
        plt.twinx()
        self.AI_gradient_Drought_year_spatial_tif(delta_tif)
        plt.tight_layout()

        # outf = join(outdir,'delta_area_ratio_bar.pdf')
        outf = join(outdir,'legend.pdf')
        plt.savefig(outf)
        plt.close()
        T.open_path_and_file(outdir)
        # plt.show()

    def delta_area_statistic(self):
        tif = join(self.this_class_tif, 'delta_hot_normal/drought_year_1.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(tif)
        df = T.spatial_dics_to_df({'delta': spatial_dict})
        # df = Dataframe_func(df).df
        df_less_than_0 = df[df['delta']<-0]
        df_more_than_0 = df[df['delta']>0]

        ratio_less_than_0 = len(df_less_than_0) / len(df) * 100
        ratio_more_than_0 = len(df_more_than_0) / len(df) * 100
        print(f'ratio_less_than_0:{ratio_less_than_0:.2f}')
        print(f'ratio_more_than_0:{ratio_more_than_0:.2f}')
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        plt.imshow(arr,cmap='RdBu',vmin=-1,vmax=1,interpolation='nearest')
        plt.show()
        exit()

    def delta_value_statistic(self):
        outdir = join(self.this_class_png, 'delta_value_statistic')
        T.mk_dir(outdir)
        outf = join(outdir,'delta_value_statistic.pdf')
        delta_tif = join(self.this_class_tif, 'spatial_two_mode_magnitude/spatial_two_mode_magnitude_1.tif')
        hot_tif = join(self.this_class_tif, 'Drought_year_spatial_tif/hot-drought.tif')
        normal_tif = join(self.this_class_tif,'Drought_year_spatial_tif/normal-drought.tif')

        delta_dict = DIC_and_TIF().spatial_tif_to_dic(delta_tif)
        hot_dict = DIC_and_TIF().spatial_tif_to_dic(hot_tif)
        normal_dict = DIC_and_TIF().spatial_tif_to_dic(normal_tif)

        all_spatial_dict = {
            'delta':delta_dict,
            'hot-drought':hot_dict,
            'normal-drought':normal_dict
        }
        df = T.spatial_dics_to_df(all_spatial_dict)
        df = Dataframe_func(df).df
        ELI_class_list = global_ELI_class_list
        result_dict = {}
        for ELI_class in ELI_class_list:
            df_ELI = df[df['ELI_class'] == ELI_class]
            delta_vals = df_ELI['delta'].tolist()
            hot_vals = df_ELI['hot-drought'].tolist()
            normal_vals = df_ELI['normal-drought'].tolist()

            delta_mean = np.nanmean(delta_vals)
            hot_mean = np.nanmean(hot_vals)
            normal_mean = np.nanmean(normal_vals)

            delta_err,_,_ = T.uncertainty_err(delta_vals)
            hot_err,_,_ = T.uncertainty_err(hot_vals)
            normal_err,_,_ = T.uncertainty_err(normal_vals)

            delta_err = abs(delta_err)
            hot_err = abs(hot_err)
            normal_err = abs(normal_err)

            result_dict[ELI_class] = {
                'delta_mean':delta_mean,
                'hot_mean':hot_mean,
                'normal_mean':normal_mean,
                'delta_err':delta_err,
                'hot_err':hot_err,
                'normal_err':normal_err,
            }
        df_result = T.dic_to_df(result_dict,'ELI_class')
        T.print_head_n(df_result)
        ELI_class_list_1 = df_result['ELI_class'].tolist()
        plt.figure(figsize=(18*centimeter_factor,4*centimeter_factor))
        plt.subplot(1,3,1)
        delta_mean_list = df_result['delta_mean'].tolist()
        delta_err_list = df_result['delta_err'].tolist()
        plt.bar(ELI_class_list_1,delta_mean_list,yerr=delta_err_list,width=0.5)
        plt.ylim(-0.5,0)
        plt.title('delta')

        plt.subplot(1,3,2)
        hot_mean_list = df_result['hot_mean'].tolist()
        hot_err_list = df_result['hot_err'].tolist()
        plt.bar(ELI_class_list_1,hot_mean_list,yerr=hot_err_list,width=0.5)
        plt.ylim(-0.5,0)
        plt.title('hot-drought')

        plt.subplot(1,3,3)
        normal_mean_list = df_result['normal_mean'].tolist()
        normal_err_list = df_result['normal_err'].tolist()
        plt.bar(ELI_class_list_1,normal_mean_list,yerr=normal_err_list,width=0.5)
        plt.ylim(-0.5,0)
        plt.title('normal-drought')

        plt.tight_layout()
        plt.savefig(outf,dpi=300)
        plt.close()
        T.open_path_and_file(outdir)

    def print_delta_value_statistic(self):
        delta_tif = join(self.this_class_tif, 'delta_hot_normal/drought_year_1.tif')
        delta_dict = DIC_and_TIF().spatial_tif_to_dic(delta_tif)
        val_list = []
        for pix in delta_dict:
            val = delta_dict[pix]
            val_list.append(val)
        plt.hist(val_list,bins=100)
        mean = np.nanmean(val_list)
        print(mean)
        plt.show()


class Compensation_Excerbation_heatwave:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Compensation_Excerbation_heatwave', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr,'dataframe/heatwave.df')
        pass

    def run(self):
        # self.copy_df()
        # df = self.__gen_df_init()
        #
        # df = self.add_variables_process(df,Load_Data().NDVI_origin)
        # df = self.add_variables_process(df,Load_Data().NDVI_anomaly_detrend)
        # df = self.add_SPI_process(df)
        # df = self.add_rt(df)
        #
        # df = Dataframe_func(df).df
        # T.save_df(df, self.dff)
        # T.df_to_excel(df, self.dff)

        # self.rt_tif()
        # self.delta_heat_drought()
        self.statistic_delta_heat_drought()
        # self.variable_trajectory()

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

    def copy_df(self):
        if isfile(self.dff):
            print('already exists: ', self.dff)
            print('press enter to overwrite')
            pause()
            pause()
            pause()
        import analysis
        outdir = join(self.this_class_arr,'dataframe')
        T.mk_dir(outdir)
        dff = join(analysis.Pick_Heatwave_Events().this_class_arr,r'heatwave_dataframe\heatwave_dataframe.df')
        df = T.load_df(dff)
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)
        pass

    def add_variables_process(self, df,variable_func):
        # df = Load_dataframe()
        NDVI_spatial_dict, var_name, valid_range = variable_func()
        year_list = global_year_range_list
        gs = global_gs
        NDVI_list_all = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            drought_year = row['hot_year']
            if not pix in NDVI_spatial_dict:
                NDVI_list_all.append(np.nan)
                continue
            NDVI = NDVI_spatial_dict[pix]
            NDVI = np.array(NDVI, dtype=float)
            NDVI[NDVI<valid_range[0]] = np.nan
            NDVI[NDVI>valid_range[1]] = np.nan
            if T.is_all_nan(NDVI):
                NDVI_list_all.append(np.nan)
                continue
            NDVI_gs = T.monthly_vals_to_annual_val(NDVI, gs, method='array')
            NDVI_gs_dict = T.dict_zip(year_list, NDVI_gs)
            NDVI_list = []
            year_list_i = []
            for y in range(-1, 5):
                y_i = drought_year + y
                if y_i in NDVI_gs_dict:
                    NDVI_list.append(NDVI_gs_dict[drought_year + y])
                else:
                    NDVI_list.append([np.nan] * len(gs))
                year_list_i.append(y_i)

            NDVI_list = np.array(NDVI_list)
            NDVI_list = NDVI_list.flatten()
            NDVI_list_all.append(NDVI_list)
        df[f'{var_name}_progress'] = NDVI_list_all
        return df

    def add_SPI_process(self, df):
        # df = Load_dataframe()
        NDVI_spatial_dict, var_name, valid_range = Load_Data().SPI_scale(3)
        year_list = global_year_range_list
        gs = global_gs
        NDVI_list_all = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            drought_year = row['hot_year']
            if not pix in NDVI_spatial_dict:
                NDVI_list_all.append(np.nan)
                continue
            NDVI = NDVI_spatial_dict[pix]
            NDVI = np.array(NDVI, dtype=float)
            NDVI[NDVI<valid_range[0]] = np.nan
            NDVI[NDVI>valid_range[1]] = np.nan
            if T.is_all_nan(NDVI):
                NDVI_list_all.append(np.nan)
                continue
            NDVI_gs = T.monthly_vals_to_annual_val(NDVI, gs, method='array')
            NDVI_gs_dict = T.dict_zip(year_list, NDVI_gs)
            NDVI_list = []
            year_list_i = []
            for y in range(-1, 5):
                y_i = drought_year + y
                if y_i in NDVI_gs_dict:
                    NDVI_list.append(NDVI_gs_dict[drought_year + y])
                else:
                    NDVI_list.append([np.nan] * len(gs))
                year_list_i.append(y_i)

            NDVI_list = np.array(NDVI_list)
            NDVI_list = NDVI_list.flatten()
            NDVI_list_all.append(NDVI_list)
        df[f'{var_name}_progress'] = NDVI_list_all
        return df

    def add_rt(self,df):
        rt_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            hot_year = row['hot_year']
            NDVI = row['NDVI-anomaly_detrend_progress']
            if type(NDVI) == float:
                rt_list.append(np.nan)
                continue
            NDVI_reshape = np.array(NDVI)
            NDVI_reshape = np.reshape(NDVI_reshape,(-1, 6))
            hot_year_NDVI = NDVI_reshape[1]
            hot_year_NDVI_mean = np.nanmean(hot_year_NDVI)
            rt_list.append(hot_year_NDVI_mean)
        df['rt'] = rt_list

        return df

    def rt_tif(self):
        outdir = join(self.this_class_tif, 'rt')
        T.mk_dir(outdir)
        dff = self.dff
        df = T.load_df(dff)
        df_pix_dict = T.df_groupby(df, 'pix')
        spatial_dict = {}
        for pix in df_pix_dict:
            df_pix = df_pix_dict[pix]
            rt_list = df_pix['rt'].tolist()
            rt_mean = np.nanmean(rt_list)
            spatial_dict[pix] = rt_mean
        outf = join(outdir, 'rt.tif')
        DIC_and_TIF().pix_dic_to_tif(spatial_dict, outf)

        pass

    def delta_heat_drought(self):
        outdir = join(self.this_class_tif, 'delta_heat_drought')
        T.mk_dir(outdir)
        drought_dir = join(Compensation_Excerbation().this_class_tif,'Drought_year_spatial_tif')
        hot_tif = join(drought_dir,'hot-drought.tif')
        normal_tif = join(drought_dir,'normal-drought.tif')
        heatwave_tif = join(self.this_class_tif,'rt/rt.tif')

        spatial_dict_hot = DIC_and_TIF().spatial_tif_to_dic(hot_tif)
        spatial_dict_normal = DIC_and_TIF().spatial_tif_to_dic(normal_tif)
        spatial_dict_heat = DIC_and_TIF().spatial_tif_to_dic(heatwave_tif)

        spatial_dict_all = {
            'hot': spatial_dict_hot,
            'normal': spatial_dict_normal,
            'heat': spatial_dict_heat
        }

        df = T.spatial_dics_to_df(spatial_dict_all)

        delta_heat_normal = []
        delta_hot_normal = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            heat = row['heat']
            normal = row['normal']
            hot = row['hot']
            delta_heat = heat - normal
            delta_hot = hot - normal
            delta_heat_normal.append(delta_heat)
            delta_hot_normal.append(delta_hot)
        df['delta_heat'] = delta_heat_normal
        df['delta_hot'] = delta_hot_normal
        T.print_head_n(df, 10)

        outf_delta_heat = join(outdir, 'delta_heat.tif')
        delta_heat_dict = T.df_to_spatial_dic(df,'delta_heat')
        DIC_and_TIF().pix_dic_to_tif(delta_heat_dict, outf_delta_heat)

        outf_delta_hot = join(outdir, 'delta_hot.tif')
        delta_hot_dict = T.df_to_spatial_dic(df,'delta_hot')
        DIC_and_TIF().pix_dic_to_tif(delta_hot_dict, outf_delta_hot)

        pass

    def statistic_delta_heat_drought(self):
        drought_dir = join(Compensation_Excerbation().this_class_tif, 'Drought_year_spatial_tif')
        hot_tif = join(drought_dir, 'hot-drought.tif')
        normal_tif = join(drought_dir, 'normal-drought.tif')
        heatwave_tif = join(self.this_class_tif, 'rt/rt.tif')

        spatial_dict_hot = DIC_and_TIF().spatial_tif_to_dic(hot_tif)
        spatial_dict_normal = DIC_and_TIF().spatial_tif_to_dic(normal_tif)
        spatial_dict_heat = DIC_and_TIF().spatial_tif_to_dic(heatwave_tif)

        spatial_dict_all = {
            'hot': spatial_dict_hot,
            'normal': spatial_dict_normal,
            'heat': spatial_dict_heat
        }

        df = T.spatial_dics_to_df(spatial_dict_all)

        delta_heat_normal = []
        delta_hot_normal = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            heat = row['heat']
            normal = row['normal']
            hot = row['hot']
            delta_heat = heat - normal
            delta_hot = hot - normal
            delta_heat_normal.append(delta_heat)
            delta_hot_normal.append(delta_hot)
        df['delta_heat'] = delta_heat_normal
        df['delta_hot'] = delta_hot_normal
        T.print_head_n(df, 10)

        KDE_plot().plot_scatter(delta_heat_normal,delta_hot_normal,s=4)
        plt.xlabel('delta_heat_normal')
        plt.ylabel('delta_hot_normal')
        plt.axis('equal')
        plt.ylim(-2,2)
        plt.show()

        pass

    def variable_trajectory(self,df):
        outdir = join(self.this_class_png, 'variable_trajectory')
        T.mk_dir(outdir)
        ELI_class_list = global_ELI_class_list
        drought_type_list = global_drought_type_list

        season_list = []

        for i,row in df.iterrows():
            mon = row['drought_mon']
            season = global_season_mon_dict[mon]
            season_list.append(season)
        df['drought_season'] = season_list

        for timing in timing_list:
            for ELI_class in ELI_class_list:
                plt.figure(figsize=(9*centimeter_factor,6*centimeter_factor))
                for drt in drought_type_list:
                    df_ELI = df[df['ELI_class'] == ELI_class]
                    df_drt = df_ELI[df_ELI['drought_type'] == drt]
                    df_timing = df_drt[df_drt['drought_season'] == timing]
                    NDVI_process = df_timing['NDVI_progress'].tolist()
                    NDVI_process = np.array(NDVI_process)
                    NDVI_process_mean = np.nanmean(NDVI_process,axis=0)
                    NDVI_process_std = np.nanstd(NDVI_process,axis=0) / 6.
                    # NDVI_process_std = T.uncertainty_err_2d(NDVI_process,axis=0)
                    NDVI_process_mean = NDVI_process_mean[:3*6]
                    NDVI_process_std = NDVI_process_std[:3*6]
                    plt.plot(NDVI_process_mean,label=f'{drt}')
                    plt.fill_between(range(len(NDVI_process_mean)),NDVI_process_mean-NDVI_process_std,NDVI_process_mean+NDVI_process_std,alpha=0.3)
                plt.legend()
                plt.title(f'{timing}\n{ELI_class}')
                plt.ylim(-1.1,0.5)
                plt.grid()
                plt.xticks(list(range(len(NDVI_process_mean)))[::6],[-1,0,1])
                fname = f'{timing}_{ELI_class}.pdf'
                outf = join(outdir,fname)
                plt.savefig(outf)
                plt.close()
                # plt.show()
        T.open_path_and_file(outdir)

        pass


class Drought_timing:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Drought_timing', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'Drought_timing.df')
        pass

    def run(self):
        # df = Over_shoot_phenology().add_NDVI_process()
        # df = self.__gen_df_init()
        # T.print_head_n(df)
        # exit()
        # df = Compensation_Excerbation().add_SM_anomaly_process(df)
        # df = Compensation_Excerbation().add_Tair_anomaly_process(df)
        # df = Compensation_Excerbation().add_Tair_origin_process(df)
        # opt_temp_f = '/Volumes/NVME2T/Energy_water_hotdrought/results/analysis/Optimal_temperature/tif/optimal_temperature/LT_Baseline_NT_origin_step_0.5_celsius_resample.tif'
        # spatial_dict = DIC_and_TIF().spatial_tif_to_dic(opt_temp_f)
        # df = T.add_spatial_dic_to_df(df,spatial_dict,'optimal_temp')
        # T.print_head_n(df)
        # exit()
        # Dataframe
        # df = self.add_drought_season(df)


        # T.save_df(df, self.dff)
        # T.df_to_excel(df, self.dff)

        # statistic
        # self.timing_trajectory(df)
        # self.timing_trajectory_sm(df)
        # self.timing_trajectory_Tair(df)
        # self.delta_season_tif(df)
        # self.delta_season_bar(df)
        # self.delta_season_bar_all(df)
        # self.delta_season_bar_all1()
        # self.delta_season_box_all(df)
        # self.delta_season_bar_ANOVA(df)
        # self.delta_season_bar_error_bar(df)
        self.check_compensation_excerbation_season()

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

    def add_drought_season(self,df):
        season_list = []
        for i, row in df.iterrows():
            mon = row['drought_mon']
            season = global_season_mon_dict[mon]
            season_list.append(season)
        df['drought_season'] = season_list
        return df

    def timing_trajectory(self,df):
        outdir = join(self.this_class_png, 'timing_trajectory')
        T.mk_dir(outdir)
        timing_list = global_drought_season_list
        ELI_class_list = global_ELI_class_list
        drought_type_list = global_drought_type_list

        season_list = []

        for i,row in df.iterrows():
            mon = row['drought_mon']
            season = global_season_mon_dict[mon]
            season_list.append(season)
        df['drought_season'] = season_list

        for timing in timing_list:
            for ELI_class in ELI_class_list:
                plt.figure(figsize=(9*centimeter_factor,6*centimeter_factor))
                for drt in drought_type_list:
                    df_ELI = df[df['ELI_class'] == ELI_class]
                    df_drt = df_ELI[df_ELI['drought_type'] == drt]
                    df_timing = df_drt[df_drt['drought_season'] == timing]
                    NDVI_process = df_timing['NDVI_progress'].tolist()
                    NDVI_process = np.array(NDVI_process)
                    NDVI_process_mean = np.nanmean(NDVI_process,axis=0)
                    NDVI_process_std = np.nanstd(NDVI_process,axis=0) / 6.
                    # NDVI_process_std = T.uncertainty_err_2d(NDVI_process,axis=0)
                    NDVI_process_mean = NDVI_process_mean[:3*6]
                    NDVI_process_std = NDVI_process_std[:3*6]
                    plt.plot(NDVI_process_mean,label=f'{drt}')
                    plt.fill_between(range(len(NDVI_process_mean)),NDVI_process_mean-NDVI_process_std,NDVI_process_mean+NDVI_process_std,alpha=0.3)
                plt.legend()
                plt.title(f'{timing}\n{ELI_class}')
                plt.ylim(-1.1,0.5)
                plt.grid()
                plt.xticks(list(range(len(NDVI_process_mean)))[::6],[-1,0,1])
                fname = f'{timing}_{ELI_class}.pdf'
                outf = join(outdir,fname)
                plt.savefig(outf)
                plt.close()
                # plt.show()
        T.open_path_and_file(outdir)

        pass

    def timing_trajectory_sm(self,df):
        outdir = join(self.this_class_png, 'timing_trajectory_sm')
        T.mk_dir(outdir)
        timing_list = global_drought_season_list
        ELI_class_list = global_ELI_class_list
        drought_type_list = global_drought_type_list

        season_list = []

        for i,row in df.iterrows():
            mon = row['drought_mon']
            season = global_season_mon_dict[mon]
            season_list.append(season)
        df['drought_season'] = season_list

        for timing in timing_list:
            for ELI_class in ELI_class_list:
                plt.figure(figsize=(9*centimeter_factor,6*centimeter_factor))
                for drt in drought_type_list:
                    df_ELI = df[df['ELI_class'] == ELI_class]
                    df_drt = df_ELI[df_ELI['drought_type'] == drt]
                    df_timing = df_drt[df_drt['drought_season'] == timing]
                    NDVI_process = df_timing['GLEAM-SMRoot-anomaly_detrend_progress'].tolist()
                    NDVI_process = np.array(NDVI_process)
                    NDVI_process_mean = np.nanmean(NDVI_process,axis=0)
                    NDVI_process_mean = NDVI_process_mean[:3*6]
                    plt.plot(NDVI_process_mean,label=f'{drt}')
                plt.legend()
                plt.title(f'{timing}\n{ELI_class}')
                plt.ylim(-2,0.7)
                plt.grid()
                plt.xticks(list(range(len(NDVI_process_mean)))[::6],[-1,0,1])
                fname = f'{timing}_{ELI_class}.pdf'
                outf = join(outdir,fname)
                plt.savefig(outf)
                plt.close()
                # plt.show()
        T.open_path_and_file(outdir)

        pass

    def timing_trajectory_Tair(self,df):
        outdir = join(self.this_class_png, 'timing_trajectory_Tair')
        T.mk_dir(outdir)
        timing_list = global_drought_season_list
        ELI_class_list = global_ELI_class_list
        drought_type_list = global_drought_type_list
        # T.print_head_n(df)
        # exit()

        season_list = []

        for i,row in df.iterrows():
            mon = row['drought_mon']
            season = global_season_mon_dict[mon]
            season_list.append(season)
        df['drought_season'] = season_list

        for timing in timing_list:
            for ELI_class in ELI_class_list:
                plt.figure(figsize=(9*centimeter_factor,6*centimeter_factor))
                for drt in drought_type_list:
                    df_ELI = df[df['ELI_class'] == ELI_class]
                    df_drt = df_ELI[df_ELI['drought_type'] == drt]
                    df_timing = df_drt[df_drt['drought_season'] == timing]
                    # NDVI_process = df_timing['Temperature-anomaly_detrend_progress'].tolist()
                    optimal_temp = df_timing['optimal_temp'].tolist()
                    optimal_temp = list(set(optimal_temp))
                    optimal_temp_mean = np.nanmean(optimal_temp)
                    NDVI_process = df_timing['Temperature-origin_progress'].tolist()
                    NDVI_process = np.array(NDVI_process)
                    NDVI_process_mean = np.nanmean(NDVI_process,axis=0)
                    NDVI_process_mean = NDVI_process_mean[:3*6]
                    plt.plot(NDVI_process_mean,label=f'{drt}')
                    plt.plot([optimal_temp_mean]*len(NDVI_process_mean),label=f'optimal_temp')
                    # plt.show()
                plt.legend()
                plt.title(f'{timing}\n{ELI_class}')
                plt.ylim(0,26)
                plt.grid()
                plt.xticks(list(range(len(NDVI_process_mean)))[::6],[-1,0,1])
                fname = f'{timing}_{ELI_class}.pdf'
                outf = join(outdir,fname)
                plt.savefig(outf)
                plt.close()
                # plt.show()
        T.open_path_and_file(outdir)

        pass

    def delta_season_tif(self,df):
        outdir = join(self.this_class_tif, 'delta_season')
        T.mk_dir(outdir)
        drought_season_list = global_drought_season_list
        for season in drought_season_list:
            df_season = df[df['drought_season'] == season]
            # print(df_season)
            pix_list = T.get_df_unique_val_list(df_season, 'pix')
            # print(len(pix_list))
            drought_type_list = global_drought_type_list

            # drought_year_list = range(1, 5)
            drought_year_list = [1]
            for drought_year_i in drought_year_list:
                df_group_dict = T.df_groupby(df_season, 'pix')
                spatial_dict = {}
                for pix in tqdm(pix_list):
                    df_pix = df_group_dict[pix]
                    df_hot = df_pix[df_pix['drought_type'] == 'hot-drought']
                    df_normal = df_pix[df_pix['drought_type'] == 'normal-drought']
                    if len(df_hot) == 0 or len(df_normal) == 0:
                        continue
                    # NDVI_progress = row['NDVI_progress'].tolist()
                    # NDVI_progress = np.array(NDVI_progress)
                    # NDVI_progress_reshape = NDVI_progress.reshape(-1, 6)
                    # drought_year_NDVI = NDVI_progress_reshape[1]
                    NDVI_progress_hot = df_hot['NDVI_progress'].tolist()
                    NDVI_progress_normal = df_normal['NDVI_progress'].tolist()
                    mean_hot = np.nanmean(NDVI_progress_hot, axis=0)
                    mean_normal = np.nanmean(NDVI_progress_normal, axis=0)

                    mean_hot_reshape = np.array(mean_hot).reshape(-1, 6)
                    mean_normal_reshape = np.array(mean_normal).reshape(-1, 6)

                    mean_hot_drought_year = mean_hot_reshape[1:drought_year_i + 1]
                    mean_normal_drought_year = mean_normal_reshape[1:drought_year_i + 1]

                    mean_normal_drought_NDVI = np.nanmean(mean_normal_drought_year)
                    mean_hot_drought_NDVI = np.nanmean(mean_hot_drought_year)
                    delta = mean_hot_drought_NDVI - mean_normal_drought_NDVI
                    spatial_dict[pix] = delta
                outf = join(outdir, f'{season}_{drought_year_i}.tif')
                DIC_and_TIF().pix_dic_to_tif(spatial_dict, outf)
        T.open_path_and_file(outdir)


    def delta_season_bar(self,df):
        outdir = join(self.this_class_png, 'delta_season_bar')
        T.mk_dir(outdir)
        timing_list = global_drought_season_list
        ELI_class_list = global_ELI_class_list
        drought_type_list = global_drought_type_list
        result_dict = {}
        drought_year_index = 1
        # drought_year_index = 2
        plt.figure(figsize=(9 * centimeter_factor, 6 * centimeter_factor))
        for timing in timing_list:
            for ELI_class in ELI_class_list:
                df_ELI = df[df['ELI_class'] == ELI_class]
                result_dict_i = {}
                for drt in drought_type_list:
                    df_drt = df_ELI[df_ELI['drought_type'] == drt]
                    df_timing = df_drt[df_drt['drought_season'] == timing]
                    NDVI_process = df_timing['NDVI_progress'].tolist()
                    NDVI_process = np.array(NDVI_process)
                    NDVI_process_mean = np.nanmean(NDVI_process, axis=0)
                    NDVI_process_mean_reshape = NDVI_process_mean.reshape(-1, 6)
                    drought_year_NDVI_process = NDVI_process_mean_reshape[drought_year_index]
                    drought_year_NDVI_process_mean = np.nanmean(drought_year_NDVI_process)
                    result_dict_i[drt] = drought_year_NDVI_process_mean
                # delta = result_dict_i['hot-drought'] - result_dict_i['normal-drought']
                delta = result_dict_i['normal-drought'] - result_dict_i['hot-drought']
                key = f'{timing}_{ELI_class}'
                result_dict[key] = delta
        print(result_dict)
        keys = list(result_dict.keys())
        values = [result_dict[key] for key in keys]
        plt.bar(keys, values, color='grey', width=0.4)
        plt.plot(keys,[0]*len(keys),'k--',linewidth=0.5)
        plt.xticks(rotation=90)
        plt.ylim(-0.5, 0.5)
        # plt.grid()
        plt.tight_layout()
        # outf = join(outdir, f'delta_season_bar_{drought_year_index}.pdf')
        # plt.savefig(outf)
        # plt.close()
        # T.open_path_and_file(outdir)
        plt.show()

    def delta_season_bar_ratio(self,df):
        outdir = join(self.this_class_png, 'delta_season_bar_ratio')
        T.mk_dir(outdir)
        timing_list = global_drought_season_list
        ELI_class_list = global_ELI_class_list
        drought_type_list = global_drought_type_list
        result_dict = {}
        drought_year_index = 1
        # drought_year_index = 2
        plt.figure(figsize=(9 * centimeter_factor, 6 * centimeter_factor))
        for timing in timing_list:
            for ELI_class in ELI_class_list:
                df_ELI = df[df['ELI_class'] == ELI_class]
                result_dict_i = {}
                for drt in drought_type_list:
                    df_drt = df_ELI[df_ELI['drought_type'] == drt]
                    df_timing = df_drt[df_drt['drought_season'] == timing]
                    NDVI_process = df_timing['NDVI_progress'].tolist()
                    NDVI_process = np.array(NDVI_process)
                    NDVI_process_mean = np.nanmean(NDVI_process, axis=0)
                    NDVI_process_mean_reshape = NDVI_process_mean.reshape(-1, 6)
                    drought_year_NDVI_process = NDVI_process_mean_reshape[drought_year_index]
                    drought_year_NDVI_process_mean = np.nanmean(drought_year_NDVI_process)
                    result_dict_i[drt] = drought_year_NDVI_process_mean
                # delta = result_dict_i['hot-drought'] - result_dict_i['normal-drought']
                delta = result_dict_i['normal-drought'] - result_dict_i['hot-drought']
                key = f'{timing}_{ELI_class}'
                result_dict[key] = delta
        print(result_dict)
        keys = list(result_dict.keys())
        values = [result_dict[key] for key in keys]
        plt.bar(keys, values, color='grey', width=0.4)
        plt.plot(keys,[0]*len(keys),'k--',linewidth=0.5)
        plt.xticks(rotation=90)
        plt.ylim(-0.5, 0.5)
        # plt.grid()
        plt.tight_layout()
        # outf = join(outdir, f'delta_season_bar_{drought_year_index}.pdf')
        # plt.savefig(outf)
        # plt.close()
        # T.open_path_and_file(outdir)
        plt.show()

    def delta_season_bar_all(self,df):
        outdir = join(self.this_class_png, 'delta_season_bar_all')
        T.mk_dir(outdir)
        timing_list = global_drought_season_list
        ELI_class_list = global_ELI_class_list
        drought_type_list = global_drought_type_list
        result_dict = {}
        drought_year_index = 1
        # drought_year_index = 2
        for timing in timing_list:
            for ELI_class in ELI_class_list:
                df_ELI = df[df['ELI_class'] == ELI_class]
                result_dict_i = {}
                for drt in drought_type_list:
                    df_drt = df_ELI[df_ELI['drought_type'] == drt]
                    df_timing = df_drt[df_drt['drought_season'] == timing]
                    NDVI_process = df_timing['NDVI_progress'].tolist()
                    NDVI_process = np.array(NDVI_process)
                    NDVI_process_mean = np.nanmean(NDVI_process, axis=0)
                    NDVI_process_mean_reshape = NDVI_process_mean.reshape(-1, 6)
                    drought_year_NDVI_process = NDVI_process_mean_reshape[drought_year_index]
                    drought_year_NDVI_process_mean = np.nanmean(drought_year_NDVI_process)
                    result_dict_i[drt] = drought_year_NDVI_process_mean
                delta = result_dict_i['hot-drought'] - result_dict_i['normal-drought']
                # delta = result_dict_i['normal-drought'] - result_dict_i['hot-drought']
                key = f'{timing}_{ELI_class}'
                # result_dict[key] = delta
                result_dict[key] = {
                    'delta':delta,
                    'normal-drought':result_dict_i['normal-drought'],
                    'hot-drought':result_dict_i['hot-drought'],
                }
        print(result_dict)
        keys = list(result_dict.keys())
        values_delta = [result_dict[key]['delta'] for key in keys]
        value_hot = [result_dict[key]['hot-drought'] for key in keys]
        value_normal = [result_dict[key]['normal-drought'] for key in keys]

        values_delta_str = [f'{value_delta:.2f}' for value_delta in values_delta]
        value_hot_str = [f'{value_hot:.2f}' for value_hot in value_hot]
        value_normal_str = [f'{value_normal:.2f}' for value_normal in value_normal]

        plt.figure(figsize=(18 * centimeter_factor, 6 * centimeter_factor))

        plt.subplot(1,3,1)
        plt.bar(keys, values_delta, color='grey', width=0.4)
        for i in range(len(keys)):
            plt.text(keys[i],values_delta[i],values_delta_str[i],ha='center',va='bottom',fontsize=8)
        # print(keys)
        plt.plot(keys,[0]*len(keys),'k--',linewidth=0.5)
        plt.xticks(rotation=90)
        plt.ylim(-0.5, 0.5)
        plt.title('delta')

        plt.subplot(1, 3, 2)
        plt.bar(keys, value_hot, color='grey', width=0.4)
        for i in range(len(keys)):
            plt.text(keys[i],value_hot[i],value_hot_str[i],ha='center',va='bottom',fontsize=8)
        plt.plot(keys,[0]*len(keys),'k--',linewidth=0.5)
        plt.xticks(rotation=90)
        plt.ylim(-1, 1)
        plt.title('hot-drought')

        plt.subplot(1, 3, 3)
        plt.bar(keys, value_normal, color='grey', width=0.4)
        for i in range(len(keys)):
            plt.text(keys[i],value_normal[i],value_normal_str[i],ha='center',va='bottom',fontsize=8)
        plt.plot(keys,[0]*len(keys),'k--',linewidth=0.5)
        plt.xticks(rotation=90)
        plt.ylim(-1, 1)
        plt.title('normal-drought')


        plt.tight_layout()


        # outf = join(outdir, f'delta_season_bar_{drought_year_index}.pdf')
        outf = join(outdir, f'delta_season_bar_{drought_year_index}_legend.pdf')
        # plt.savefig(outf)
        # plt.close()
        # T.open_path_and_file(outdir)
        plt.show()

    def delta_season_bar_all1(self):
        fdir = join(Random_Forests_delta().this_class_arr, 'seasonal_delta')
        outdir = join(self.this_class_png, 'delta_season_bar_all')

        season_list = global_drought_season_list
        ELI_class_list = global_ELI_class_list

        box_list = []
        mean_list = []
        err_list = []
        label_list = []
        for season in season_list:
            fpath = join(fdir,f'{season}.df')
            df = T.load_df(fpath)
            # plt.figure()
            for ELI_class in ELI_class_list:
                df_ELI = df[df['ELI_class'] == ELI_class]
                NDVI_delta = df_ELI['NDVI_delta'].tolist()
                mean = np.nanmean(NDVI_delta)
                # err = np.nanstd(NDVI_delta)
                err = T.uncertainty_err(NDVI_delta)[0]
                err = abs(err)
                mean_list.append(mean)
                err_list.append(err)
                box_list.append(NDVI_delta)
                label = f'{season}_{ELI_class}'
                label_list.append(label)
        # plt.boxplot(box_list,labels=label_list,showfliers=False)
        plt.barh(label_list,mean_list,xerr=err_list)
        plt.xticks(rotation=90)
        # plt.ylim(-3,3)
        plt.tight_layout()
        outf1 = join(outdir, f'delta_season_bar_all_ratio1.pdf')
        plt.savefig(outf1)
        plt.close()

        threshold_list = [-np.inf, -0.5, -0.25, 0, 0.25, 0.5, np.inf]
        color_list = T.gen_colors(len(threshold_list))
        color_list_neg = T.gen_colors(3,palette='Reds')
        color_list_pos = T.gen_colors(3,palette='Blues')
        plt.figure()
        for season in season_list:
            fpath = join(fdir,f'{season}.df')
            df = T.load_df(fpath)
            for ELI_class in ELI_class_list:
                df_ELI = df[df['ELI_class'] == ELI_class]
                ratio_list = []
                bottom_neg = 0
                bottom_pos = 0
                bottom = 0
                x = f'{season}_{ELI_class}'
                ratio_neg_list = []
                ratio_pos_list = []
                for i in range(len(threshold_list)):
                    if i + 1 == len(threshold_list):
                        break
                    threshold_left = threshold_list[i]
                    threshold_right = threshold_list[i + 1]
                    df_ELI_threshold = df_ELI[df_ELI['NDVI_delta'] < threshold_right]
                    df_ELI_threshold = df_ELI_threshold[df_ELI_threshold['NDVI_delta'] > threshold_left]
                    ratio = len(df_ELI_threshold) / len(df_ELI) * 100
                    print(threshold_left)
                    if threshold_left < 0:
                        ratio_neg = -ratio
                        ratio_neg_list.append(ratio_neg+bottom_neg)

                    else:
                        ratio_pos = ratio
                        ratio_pos_list.append(ratio_pos)
                        # plt.bar(x,ratio_pos,bottom=bottom_pos,label=f'{threshold_left}~{threshold_right}',color=color_list[i])
                        # bottom_pos = bottom_pos + ratio_pos
                    # plt.bar(x,ratio,bottom=bottom,label=f'{threshold_left}~{threshold_right}',color=color_list[i])
                    # bottom = bottom + ratio
                    # ratio_list.append(ratio)
                # plt.bar(x, ratio_pos_list, bottom=bottom_pos, label=f'{threshold_left}~{threshold_right}', color='r')
                # print('---')
                ratio_neg_list = ratio_neg_list[::-1]
                for i in range(len(ratio_neg_list)):
                    plt.barh(x, ratio_neg_list[i], left=bottom_neg, label=f'{threshold_list[i]}~{threshold_list[i+1]}', color=color_list_neg[i])
                    bottom_neg = bottom_neg + ratio_neg_list[i]
                for i in range(len(ratio_pos_list)):
                    plt.barh(x, ratio_pos_list[i], left=bottom_pos, label=f'{threshold_list[i]}~{threshold_list[i+1]}', color=color_list_pos[i])
                    bottom_pos = bottom_pos + ratio_pos_list[i]
            # plt.legend()
            plt.title(f'{season}')
            plt.xlim(-100,100)
            plt.tight_layout()
        outf2 = join(outdir, f'delta_season_bar_all_ratio2.pdf')
        plt.savefig(outf2)
        plt.close()
        # plt.show()
        T.open_path_and_file(outdir)


    def delta_season_box_all(self,df):
        outdir = join(self.this_class_png, 'delta_season_box_all')
        T.mk_dir(outdir)
        timing_list = global_drought_season_list
        ELI_class_list = global_ELI_class_list
        drought_type_list = global_drought_type_list
        result_dict = {}
        drought_year_index = 1
        # drought_year_index = 2
        box_list = []
        x_list = []
        for timing in timing_list:

            for ELI_class in ELI_class_list:
                df_ELI = df[df['ELI_class'] == ELI_class]
                result_dict_i = {}
                for drt in drought_type_list:
                    df_drt = df_ELI[df_ELI['drought_type'] == drt]
                    df_timing = df_drt[df_drt['drought_season'] == timing]
                    NDVI_process = df_timing['NDVI_progress'].tolist()
                    NDVI_process = np.array(NDVI_process)
                    drought_year_values = []
                    for i in range(len(NDVI_process)):
                        NDVI_process_i = NDVI_process[i]
                        NDVI_process_i_reshape = np.array(NDVI_process_i).reshape(-1,6)
                        drought_year_mean = np.nanmean(NDVI_process_i_reshape[drought_year_index])
                        drought_year_values.append(drought_year_mean)
                    key = f'{timing}_{ELI_class}_{drt}'
                    # result_dict_i[key] = drought_year_values
                    box_list.append(drought_year_values)
                    x_list.append(key)
            # plt.figure(figsize=(9 * centimeter_factor, 6 * centimeter_factor))
        plt.figure()
        plt.boxplot(box_list,labels=x_list,showfliers=False,vert=False)
        plt.xticks(rotation=90)
        # plt.ylim(-1, 1)
        # plt.title(f'{timing}')
        plt.tight_layout()
        plt.show()


    def delta_season_bar_ANOVA(self,df):
        timing_list = global_drought_season_list
        ELI_class_list = global_ELI_class_list
        drought_type_list = global_drought_type_list
        result_dict = {}
        # drought_year_index = 1
        drought_year_index = 2
        for timing in timing_list:
            for ELI_class in ELI_class_list:
                df_ELI = df[df['ELI_class'] == ELI_class]
                result_dict_i = {}
                for drt in drought_type_list:
                    df_drt = df_ELI[df_ELI['drought_type'] == drt]
                    df_timing = df_drt[df_drt['drought_season'] == timing]
                    NDVI_process = df_timing['NDVI_progress'].tolist()
                    NDVI_process_i_reshape_drought_year_mean_list = []
                    for i in range(len(NDVI_process)):
                        NDVI_process_i = NDVI_process[i]
                        NDVI_process_i_reshape = np.array(NDVI_process_i).reshape(-1, 6)
                        NDVI_process_i_reshape_drought_year = NDVI_process_i_reshape[drought_year_index]
                        NDVI_process_i_reshape_drought_year_mean = np.nanmean(NDVI_process_i_reshape_drought_year)
                        NDVI_process_i_reshape_drought_year_mean_list.append(NDVI_process_i_reshape_drought_year_mean)
                    NDVI_process_i_reshape_drought_year_mean_list = np.array(NDVI_process_i_reshape_drought_year_mean_list)
                    NDVI_process_i_reshape_drought_year_mean_list = T.remove_np_nan(NDVI_process_i_reshape_drought_year_mean_list)
                    key = f'{timing}_{ELI_class}_{drt}'
                    result_dict[key] = NDVI_process_i_reshape_drought_year_mean_list
                    # print(NDVI_process_i_reshape_drought_year_mean_list)
                    # exit()
        # print(result_dict)
        # exit()
        # 'normal-drought', 'hot-drought'
        for timing in timing_list:
            for ELI_class in ELI_class_list:
                key1 = f'{timing}_{ELI_class}_normal-drought'
                key2 = f'{timing}_{ELI_class}_hot-drought'
                f,p = f_oneway(result_dict[key1],result_dict[key2])
                print(f'{key1} vs {key2} f={f},p={p}')

    def delta_season_bar_error_bar(self,df):
        outdir = join(self.this_class_png, 'delta_season_bar_error_bar')
        T.mk_dir(outdir)
        timing_list = global_drought_season_list
        ELI_class_list = global_ELI_class_list
        drought_type_list = global_drought_type_list
        result_dict = {}
        result_dict_err = {}
        # drought_year_index = 1
        drought_year_index = 2
        plt.figure(figsize=(9 * centimeter_factor, 6 * centimeter_factor))
        for timing in timing_list:
            for ELI_class in ELI_class_list:
                df_ELI = df[df['ELI_class'] == ELI_class]
                result_dict_i = {}
                for drt in drought_type_list:
                    df_drt = df_ELI[df_ELI['drought_type'] == drt]
                    df_timing = df_drt[df_drt['drought_season'] == timing]
                    NDVI_process = df_timing['NDVI_progress'].tolist()
                    mean_list = []
                    for NDVI_process_i in NDVI_process:
                        NDVI_process_i_reshape = np.array(NDVI_process_i).reshape(-1, 6)
                        NDVI_process_i_reshape_drought_year = NDVI_process_i_reshape[drought_year_index]
                        NDVI_process_i_reshape_drought_year_mean = np.nanmean(NDVI_process_i_reshape_drought_year)
                        mean_list.append(NDVI_process_i_reshape_drought_year_mean)
                    result_dict_i[drt] = mean_list
                vals_hotdrought = result_dict_i['hot-drought']
                vals_normaldrought = result_dict_i['normal-drought']
                delta_list = []
                for i in tqdm(range(10000),desc=f'{timing}_{ELI_class}'):
                    vals_choice_hot_drought = np.random.choice(vals_hotdrought, int(len(vals_hotdrought)*0.5))
                    vals_choice_normal_drought = np.random.choice(vals_normaldrought, int(len(vals_normaldrought)*0.5))
                    vals_choice_hot_drought_mean = np.nanmean(vals_choice_hot_drought)
                    vals_choice_normal_drought_mean = np.nanmean(vals_choice_normal_drought)
                    delta_i = vals_choice_hot_drought_mean - vals_choice_normal_drought_mean
                    delta_list.append(delta_i)
                delta_mean = np.nanmean(delta_list)
                delta_std = np.nanstd(delta_list)
                key = f'{timing}_{ELI_class}'
                result_dict[key] = delta_mean
                result_dict_err[key] = delta_std
        keys = list(result_dict.keys())
        values = [result_dict[key] for key in keys]
        values_err = [result_dict_err[key] for key in keys]
        plt.bar(keys, values, color='grey', width=0.4, yerr=values_err, capsize=2)
        plt.plot(keys, [0] * len(keys), 'k--', linewidth=0.5)
        plt.xticks(rotation=90)
        plt.ylim(-0.5, 0.5)
        # plt.grid()
        plt.tight_layout()
        outf = join(outdir, f'delta_season_bar_{drought_year_index}.pdf')
        # plt.show()
        plt.savefig(outf)
        plt.close()
        T.open_path_and_file(outdir)

    def check_compensation_excerbation_season(self):
        df = T.load_df(self.dff)
        T.print_head_n(df)

        ELI_class_list = global_ELI_class_list
        drought_type = global_drought_type_list
        # rt > 1
        df = df[df['rt'] > 1]
        df = df[df['drought_season'] != 'other']

        for drt in drought_type:
            df_drt = df[df['drought_type'] == drt]
            plt.figure()
            for ELI_class in ELI_class_list:
                df_ELI = df_drt[df_drt['ELI_class'] == ELI_class]
                drought_season = df_ELI['drought_season'].tolist()
                drought_season_set = set(drought_season)
                drought_season_set = list(drought_season_set)
                drought_season_set.sort()
                count = []
                x = []
                for season in drought_season_set:
                    count_i = drought_season.count(season)
                    count.append(count_i)
                    x.append(f'{season} {ELI_class}')
                plt.bar(x, count)
            plt.title(drt)
        plt.show()

        pass

class Random_Forests:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Random_Forests', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'Dataframe.df')
        pass

    def run(self):
        df = Over_shoot_phenology().add_NDVI_process()
        df = self.__gen_df_init()
        # df = self.gen_Y(df)
        # df = self.add_drought_season(df)
        df = self.add_SOS(df)
        # df = self.add_EOS(df)
        # df = self.add_seasonal_Temperature(df)
        # df = self.add_seasonal_Precipitation(df)
        # df = self.add_seasonal_SPI(df)
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)

        ##############################
        # self.random_forest_train(df)
        # self.plot_rf_result()
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

    def gen_Y(self,df):
        Y_list = []
        drought_year_index = 1
        for i,row in tqdm(df.iterrows(),total=len(df),desc='gen_Y'):
            NDVI_progress = row['NDVI_progress']
            NDVI_progress_reshape = np.array(NDVI_progress).reshape(-1,6)
            NDVI_progress_reshape_drought_year = NDVI_progress_reshape[drought_year_index]
            NDVI_progress_reshape_drought_year_mean = np.nanmean(NDVI_progress_reshape_drought_year)
            Y_list.append(NDVI_progress_reshape_drought_year_mean)
        df['Y'] = Y_list
        return df

    def add_drought_season(self,df):
        season_list = []
        for i, row in df.iterrows():
            mon = row['drought_mon']
            season = global_season_mon_dict[mon]
            season_list.append(season)
        df['season'] = season_list
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

    def add_EOS(self,df):
        import analysis
        pheno_str = 'late_end'
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
        df['EOS'] = pheno_val_list
        return df

    def add_seasonal_Temperature(self,df):
        # seasonal temperature
        temperature_spatial_dict = Meta_information().load_data('Temperature-anomaly_detrend')
        season_mon_dict = global_drought_season_dict
        year_list = global_year_range_list
        for season in season_mon_dict:
            season_mon_list = season_mon_dict[season]
            season_mon_index_list = np.array(season_mon_list) - 1
            season_val_list = []
            for i,row in tqdm(df.iterrows(),total=len(df),desc='add_seasonal_Temperature'):
                pix = row['pix']
                drought_year = row['drought_year']
                vals = temperature_spatial_dict[pix]
                vals_reshape = np.array(vals).reshape(-1,12)
                vals_reshape_dict = dict(zip(year_list,vals_reshape))
                drought_year_vals = vals_reshape_dict[drought_year]
                seasonal_vals = drought_year_vals[season_mon_index_list]
                seasonal_vals_mean = np.nanmean(seasonal_vals)
                season_val_list.append(seasonal_vals_mean)
            df['Temperature-anomaly_'+season] = season_val_list
        return df

    def add_seasonal_Precipitation(self,df):
        # seasonal precipitation
        precip_spatial_dict = Meta_information().load_data('Precipitation-anomaly_detrend')
        season_mon_dict = global_drought_season_dict
        year_list = global_year_range_list
        for season in season_mon_dict:
            season_mon_list = season_mon_dict[season]
            season_mon_index_list = np.array(season_mon_list) - 1
            season_val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df), desc='add_seasonal_Precipitation'):
                pix = row['pix']
                drought_year = row['drought_year']
                vals = precip_spatial_dict[pix]
                vals_reshape = np.array(vals).reshape(-1, 12)
                vals_reshape_dict = dict(zip(year_list, vals_reshape))
                drought_year_vals = vals_reshape_dict[drought_year]
                seasonal_vals = drought_year_vals[season_mon_index_list]
                seasonal_vals_mean = np.nanmean(seasonal_vals)
                season_val_list.append(seasonal_vals_mean)
            df['Precipitation-anomaly_' + season] = season_val_list
        return df
        pass

    def add_seasonal_SPI(self,df):
        # seasonal SPI
        precip_spatial_dict = Meta_information().load_data('SPI')
        season_mon_dict = global_drought_season_dict
        year_list = global_year_range_list
        for season in season_mon_dict:
            season_mon_list = season_mon_dict[season]
            season_mon_index_list = np.array(season_mon_list) - 1
            season_val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df), desc='add_seasonal_SPI'):
                pix = row['pix']
                drought_year = row['drought_year']
                vals = precip_spatial_dict[pix]
                vals_reshape = np.array(vals).reshape(-1, 12)
                vals_reshape_dict = dict(zip(year_list, vals_reshape))
                drought_year_vals = vals_reshape_dict[drought_year]
                seasonal_vals = drought_year_vals[season_mon_index_list]
                seasonal_vals_mean = np.nanmean(seasonal_vals)
                season_val_list.append(seasonal_vals_mean)
            df['SPI_' + season] = season_val_list
        return df

    def random_forest_train(self,df):
        outdir = join(self.this_class_arr,'random_forest')
        T.mk_dir(outdir,force=True)
        # 'early', 'mid', 'late'
        variable_list_dict = {
            'early': ['SOS', 'Precipitation-anomaly_early', 'Temperature-anomaly_early', 'SPI_early',],
            'mid': ['SOS', 'Precipitation-anomaly_early', 'Temperature-anomaly_early', 'Precipitation-anomaly_mid', 'Temperature-anomaly_mid'],
            'late': ['EOS', 'Precipitation-anomaly_late', 'Temperature-anomaly_late', 'SPI_late',]
        }
        print(df.columns.tolist())

        drought_type_list = global_drought_type_list
        ELI_class_list = global_ELI_class_list

        for drought_type in drought_type_list:
            for ELI_class in ELI_class_list:
                for season in variable_list_dict:
                    print('train',drought_type,ELI_class,season)
                    df_drt = df[df['drought_type']==drought_type]
                    df_ELI = df_drt[df_drt['ELI_class']==ELI_class]
                    df_season = df_ELI[df_ELI['season']==season]
                    Y = df_season['Y'].tolist()
                    variable_list = variable_list_dict[season]
                    df_new = df_season[variable_list]
                    df_new['Y'] = Y
                    df_new = df_new.dropna()
                    X = df_new[variable_list]
                    Y_new = df_new['Y']
                    # T.print_head_n(df_new)
                    clf, importances_dic, mse, r_model, score, Y_test, y_pred = self._random_forest_train(X,Y_new,variable_list)
                    outf = join(outdir,f'{drought_type}_{ELI_class}_{season}.npy')
                    result_dict = {
                        'importances_dic':importances_dic,
                        'mse':mse,
                        'r_model':r_model,
                        'score':score,
                        'Y_test':Y_test,
                        'y_pred':y_pred
                    }
                    T.save_npy(result_dict,outf)


    def _random_forest_train(self, X, Y, variable_list):
        '''
        :param X: a dataframe of x variables
        :param Y: a dataframe of y variable
        :param variable_list: a list of x variables
        :return: details of the random forest model and the importance of each variable
        '''
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) # split the data into training and testing
        clf = RandomForestRegressor(n_estimators=100, n_jobs=7) # build a random forest model
        clf.fit(X_train, Y_train) # train the model
        result = permutation_importance(clf, X_train, Y_train, scoring=None,
                                        n_repeats=50, random_state=1,
                                        n_jobs=7) # calculate the importance of each variable using permutation importance
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


    def plot_rf_result(self):
        fdir = join(self.this_class_arr,'random_forest')
        outdir = join(self.this_class_png,'random_forest')
        T.mk_dir(outdir,force=True)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            result_dict = T.load_npy(fpath)
            importances_dic = result_dict['importances_dic']
            r_model = result_dict['r_model']
            score = result_dict['score']
            title = f'{f}\nR^2={score}, r={r_model}'
            x = importances_dic.keys()
            y = [importances_dic[key] for key in x]
            plt.figure(figsize=(10,5))
            plt.bar(x,y)
            plt.title(title)
            outf = join(outdir,f'{f}.pdf')
            plt.savefig(outf,dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

class Random_Forests_delta:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Random_Forests_delta', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'Dataframe.df')
        pass

    def run(self):
        # self.gen_df()
        # ---------------------------
        # df = self.__gen_df_init()
        # df = self.add_drought_season(df)
        # T.save_df(df, self.dff)
        # T.df_to_excel(df, self.dff)

        # df = self.add_SOS(df)
        # T.save_df(df, self.dff)
        # T.df_to_excel(df, self.dff)

        # df = self.add_EOS(df)
        # T.save_df(df, self.dff)
        # T.df_to_excel(df, self.dff)

        # data_obj_list = [
        #     Load_Data().Temperature_anomaly_detrend,
        #     Load_Data().Precipitation_anomaly_detrend,
        #     Load_Data().SPI,
        #     Load_Data().Srad_anomaly_detrend,
        #     Load_Data().FAPAR_anomaly_detrend,
        # ]
        # for data_obj in data_obj_list:
        #     df = self.add_seasonal_variables(df,data_obj)
        #     T.save_df(df, self.dff)
        #     T.df_to_excel(df, self.dff)

        # self.add_seasonal_SPI(df)
        # T.save_df(df, self.dff)
        # T.df_to_excel(df, self.dff)

        # df = self.add_GS_Temperature(df)
        # T.save_df(df, self.dff)
        # T.df_to_excel(df, self.dff)

        # df = self.add_GS_Precipitation(df)
        # df = self.add_GS_SPI(df)

        # df = self.add_GS_NDVI(df)


        # T.save_df(df, self.dff)
        # T.df_to_excel(df, self.dff)

        ##############################
        self.gen_delta(df)
        # self.merge_dataframe()
        # self.check_delta()
        self.check_delta_pdf()

        # self.random_forest_train()
        # self.plot_rf_result()
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

    def gen_df(self):
        print('Warning: this function will overwrite the dataframe')
        print('Warning: this function will overwrite the dataframe')
        print('Warning: this function will overwrite the dataframe')
        pause()
        pause()
        drought_events_dff = Dataframe().dff
        drought_events_df = T.load_df(drought_events_dff)
        T.save_df(drought_events_df,self.dff)
        T.df_to_excel(drought_events_df,self.dff)


    def gen_delta(self,df):
        outdir = join(self.this_class_arr,'seasonal_delta')
        T.mk_dir(outdir,force=True)
        drought_season_list = global_drought_season_list
        var_list = [
            'SOS',
            'EOS',
            'Temperature-anomaly_detrend',
            'Precipitation-anomaly_detrend',
            'SPI',
            'Radiation-anomaly_detrend',
            'FAPAR-anomaly_detrend',
            # 'Temperature-anomaly_juping_detrend'
        ]
        spi_list = global_all_spi_list
        for scale in spi_list:
            df_scale = df[df['drought_scale']==scale]
            outdir_i = join(outdir,scale)
            T.mk_dir(outdir_i,force=True)
            for season in drought_season_list:
                df_season = df_scale[df_scale['drought_season']==season]
                df_pix_group = T.df_groupby(df_season,'pix')
                result_dict = {}
                for pix in tqdm(df_pix_group,desc=f'{scale}_{season}'):
                    df_pix = df_pix_group[pix]
                    drought_type_list = T.get_df_unique_val_list(df_pix,'drought_type')
                    if len(drought_type_list) <= 1:
                        continue
                    df_drt_group = T.df_groupby(df_pix,'drought_type')
                    NDVI_mean_dict = {}
                    Temperature_nomaly_mean_dict = {}
                    for drt in df_drt_group:
                        df_drt = df_drt_group[drt]
                        NDVI = df_drt['NDVI-anomaly_detrend_GS'].tolist()
                        if len(NDVI) == 0:
                            continue
                        NDVI_mean = np.nanmean(NDVI)
                        NDVI_mean_dict[drt] = NDVI_mean

                        T_anomaly = df_drt['Temperature-anomaly_detrend_GS'].tolist()
                        if len(T_anomaly) == 0:
                            continue
                        T_anomaly_mean = np.nanmean(T_anomaly)
                        Temperature_nomaly_mean_dict[drt] = T_anomaly_mean

                    NDVI_delta = NDVI_mean_dict['hot-drought'] - NDVI_mean_dict['normal-drought']
                    Temperature_nomaly_delta = Temperature_nomaly_mean_dict['hot-drought'] - Temperature_nomaly_mean_dict['normal-drought']

                    delta_dict = {}
                    for var_i in var_list:
                        var_mean_dict = {}
                        for drt in df_drt_group:
                            df_drt = df_drt_group[drt]
                            if var_i in ['SOS','EOS']:
                                col_name = var_i
                            else:
                                col_name = f'{var_i}_{season}'
                            val = df_drt[col_name].tolist()
                            if len(val) == 0:
                                continue
                            if T.is_all_nan(val):
                                continue
                            var_mean = np.nanmean(val)
                            var_mean_dict[drt] = var_mean
                        if len(var_mean_dict) <= 1:
                            var_delta = np.nan
                        else:
                            var_delta = var_mean_dict['hot-drought'] - var_mean_dict['normal-drought']
                        delta_dict[var_i] = var_delta
                    result_dict[pix] = {
                        'drought_season':season,
                        'lon':df_pix['lon'].tolist()[0],
                        'lat':df_pix['lat'].tolist()[0],
                        # 'landcover_GLC':df_pix['landcover_GLC'].tolist()[0],
                        'aridity_index':df_pix['aridity_index'].tolist()[0],
                        'ELI':df_pix['ELI'].tolist()[0],
                        'ELI_class':df_pix['ELI_class'].tolist()[0],
                        # 'AI_class':df_pix['AI_class'].tolist()[0],
                        'NDVI_delta':NDVI_delta,
                        'Temperature_anomaly_GS_delta':Temperature_nomaly_delta,
                    }
                    for var_i in delta_dict:
                        result_dict[pix][f'{var_i}_delta'] = delta_dict[var_i]
                    # print(result_dict)
                    # exit()
                df_season_delta = T.dic_to_df(result_dict,key_col_str='pix')
                outf = join(outdir_i,f'{season}.df')
                T.save_df(df_season_delta,outf)
                T.df_to_excel(df_season_delta,outf)

    def merge_dataframe(self):
        fdir = join(self.this_class_arr,'seasonal_delta')
        outdir = join(self.this_class_arr,'seasonal_delta_merge')
        T.mk_dir(outdir,force=True)
        season_list = global_drought_season_list

        for season in season_list:
            print(season)
            df_list = []
            length = 0
            for scale in T.listdir(fdir):
                fpath = join(fdir,scale,f'{season}.df')
                df = T.load_df(fpath)
                df['scale'] = scale
                df_list.append(df)
                length += len(df)
            df_merge = pd.concat(df_list)
            outf = join(outdir,f'{season}.df')
            T.save_df(df_merge,outf)
            T.df_to_excel(df_merge,outf)



    def check_delta(self):

        fdir = join(self.this_class_arr,'seasonal_delta')
        outdir_tif = join(self.this_class_tif,'seasonal_delta')
        outdir = join(self.this_class_png,'seasonal_delta')
        T.mk_dir(outdir_tif,force=True)
        T.mk_dir(outdir,force=True)
        for season in global_drought_season_list:
            fpath = join(fdir,f'{season}.df')
            df = T.load_df(fpath)
            spatial_dict = T.df_to_spatial_dic(df,'NDVI_delta')
            outf = join(outdir_tif,f'{season}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict,outf)

        for season in global_drought_season_list:
            fpath = join(outdir_tif,f'{season}.tif')
            Plot().plot_ortho(fpath,cmap='RdBu',vmin=-1,vmax=1)
            outf = join(outdir,f'{season}.png')
            plt.savefig(outf,dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

    def check_delta_pdf(self):
        fdir = join(self.this_class_arr,'seasonal_delta_merge')
        variables_list = [
            'FAPAR-anomaly_detrend_delta',
            'NDVI_delta',
            'Precipitation-anomaly_detrend_delta',
            'Radiation-anomaly_detrend_delta',
            'SOS_delta',
            'EOS_delta',
            'SPI_delta',
            'Temperature-anomaly_detrend_delta',
            'Temperature_anomaly_GS_delta',
        ]
        result_dict = {}
        for f in T.listdir(fdir):
            if not f.endswith('.df'):
                continue
            season = f.split('.')[0]
            fpath = join(fdir,f)
            df = T.load_df(fpath)
            # print(df.columns.tolist())
            # exit()
            result_dict_i = {}
            for var_i in variables_list:
                vals = df[var_i].tolist()
                x,y = Plot().plot_hist_smooth(vals,bins=100)
                plt.close()
                result_dict_i[var_i] = {
                    'x':x,
                    'y':y
                }
            result_dict[season] = result_dict_i
        for var_i in variables_list:
            print(var_i)
            plt.figure()
            for season in result_dict:
                x,y = result_dict[season][var_i]['x'],result_dict[season][var_i]['y']
                plt.plot(x,y,label=season)
            plt.legend()
            plt.title(var_i)
        plt.show()

    def delta_threshold(self):
        variables_threshold_dict = {
            'FAPAR-anomaly_detrend_delta':(-0.5,0.5),
            'NDVI_delta':(-2,2),
            'Precipitation-anomaly_detrend_delta':(-2,2),
            'Radiation-anomaly_detrend_delta':(-2,2),
            'SOS_delta':(-15,15),
            'EOS_delta':(-15,15),
            'SPI_delta':(-2,2),
            'Temperature-anomaly_detrend_delta':(-.5,2),
            'Temperature_anomaly_GS_delta':(0,1.5),
        }
        return variables_threshold_dict

    def add_drought_season(self,df):
        season_list = []
        for i, row in tqdm(df.iterrows(),total=len(df),desc='add_drought_season'):
            mon = row['drought_mon']
            season = global_season_mon_dict[mon]
            season_list.append(season)
        df['drought_season'] = season_list
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

    def add_SOS(self,df):
        import analysis
        pheno_str = 'early_start'
        fpath = join(analysis.Phenology().this_class_arr, 'phenology_df/phenology_df.df')
        phenology_df = T.load_df(fpath)
        cols = list(phenology_df.columns)
        print(cols)
        pheno_spatial_dict = {}
        for i, row in tqdm(phenology_df.iterrows(),total=len(phenology_df)):
            pix = row['pix']
            early_start = row[pheno_str]
            early_start_dict = dict(early_start)
            phenology_anomaly_dict = self.phenology_anomaly_dict(early_start_dict)
            pheno_spatial_dict[pix] = phenology_anomaly_dict
        pheno_val_list = []
        for i,row in tqdm(df.iterrows(),total=len(df),desc='add_SOS'):
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

    def add_EOS(self,df):
        import analysis
        pheno_str = 'late_end'
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
        for i,row in tqdm(df.iterrows(),total=len(df),desc='add_EOS'):
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
        df['EOS'] = pheno_val_list
        return df

    def add_seasonal_variables(self,df,data_obj):
        # seasonal temperature
        # temperature_spatial_dict,var_name = Load_Data().Temperature_anomaly_detrend
        pix_list = T.get_df_unique_val_list(df,'pix')
        # print(len(pix_list))
        # exit()
        data_dict,var_name = data_obj()
        season_mon_dict = global_drought_season_dict
        year_list = global_year_range_list

        seasonal_data_dict = {}
        for pix in tqdm(pix_list,desc='reshape_data_spatial_dict'):
            vals = data_dict[pix]
            vals_reshape = np.array(vals).reshape(-1, 12)
            vals_reshape_dict = dict(zip(year_list, vals_reshape))
            seasonal_data_dict_i = {}
            for season in season_mon_dict:
                season_mon_list = season_mon_dict[season]
                season_mon_index_list = np.array(season_mon_list) - 1
                seasonal_data_dict_j = {}
                for year_indx in range(len(vals_reshape)):
                    year = year_list[year_indx]
                    drought_year_vals = vals_reshape_dict[year]
                    seasonal_vals = drought_year_vals[season_mon_index_list]
                    seasonal_vals_mean = np.nanmean(seasonal_vals)
                    seasonal_data_dict_j[year] = seasonal_vals_mean
                seasonal_data_dict_i[season] = seasonal_data_dict_j
            seasonal_data_dict[pix] = seasonal_data_dict_i

        spring_vals_list = []
        summer_vals_list = []
        autumn_vals_list = []

        for i,row in tqdm(df.iterrows(),total=len(df),desc=f'add_{var_name}_seasonal_variables'):
            pix = row['pix']
            drought_year = row['drought_year']
            spring_vals = seasonal_data_dict[pix]['spring'][drought_year]
            summer_vals = seasonal_data_dict[pix]['summer'][drought_year]
            autumn_vals = seasonal_data_dict[pix]['autumn'][drought_year]
            spring_vals_list.append(spring_vals)
            summer_vals_list.append(summer_vals)
            autumn_vals_list.append(autumn_vals)
        df[f'{var_name}_spring'] = spring_vals_list
        df[f'{var_name}_summer'] = summer_vals_list
        df[f'{var_name}_autumn'] = autumn_vals_list

        return df

    def add_seasonal_SPI(self,df):
        SPI_dict_all, var_name = Load_Data().SPI()
        pix_list = T.get_df_unique_val_list(df,'pix')
        season_mon_dict = global_drought_season_dict
        year_list = global_year_range_list
        scale_seasonal_data_dict = {}
        for scale in SPI_dict_all:
            SPI_dict = SPI_dict_all[scale]
            seasonal_data_dict = {}
            for pix in tqdm(pix_list,desc=scale):
                vals = SPI_dict[pix]
                vals_reshape = np.array(vals).reshape(-1, 12)
                vals_reshape_dict = dict(zip(year_list, vals_reshape))
                seasonal_data_dict_i = {}
                for season in season_mon_dict:
                    season_mon_list = season_mon_dict[season]
                    season_mon_index_list = np.array(season_mon_list) - 1
                    seasonal_data_dict_j = {}
                    for year_indx in range(len(vals_reshape)):
                        year = year_list[year_indx]
                        drought_year_vals = vals_reshape_dict[year]
                        seasonal_vals = drought_year_vals[season_mon_index_list]
                        seasonal_vals_mean = np.nanmean(seasonal_vals)
                        seasonal_data_dict_j[year] = seasonal_vals_mean
                    seasonal_data_dict_i[season] = seasonal_data_dict_j
                seasonal_data_dict[pix] = seasonal_data_dict_i
            scale_seasonal_data_dict[scale] = seasonal_data_dict

        spring_vals_list = []
        summer_vals_list = []
        autumn_vals_list = []

        for i,row in tqdm(df.iterrows(),total=len(df),desc=f'add_{var_name}_seasonal_variables'):
            pix = row['pix']
            drought_year = row['drought_year']
            drought_scale = row['drought_scale']
            spring_vals = scale_seasonal_data_dict[drought_scale][pix]['spring'][drought_year]
            summer_vals = scale_seasonal_data_dict[drought_scale][pix]['summer'][drought_year]
            autumn_vals = scale_seasonal_data_dict[drought_scale][pix]['autumn'][drought_year]
            spring_vals_list.append(spring_vals)
            summer_vals_list.append(summer_vals)
            autumn_vals_list.append(autumn_vals)
        df[f'{var_name}_spring'] = spring_vals_list
        df[f'{var_name}_summer'] = summer_vals_list
        df[f'{var_name}_autumn'] = autumn_vals_list

        return df

    def add_GS_Temperature(self,df):
        spatial_dict,var_name = Load_Data().Temperature_anomaly_detrend()
        year_list = global_year_range_list
        gs_mon_index = np.array(global_gs,dtype=int) - 1
        gs_vals_list = []
        for i,row in tqdm(df.iterrows(),total=len(df),desc='add_GS_Temperature'):
            pix = row['pix']
            drought_year = row['drought_year']
            vals = spatial_dict[pix]
            vals_reshape = np.array(vals).reshape(-1,12)
            vals_reshape_dict = dict(zip(year_list,vals_reshape))
            drought_year_vals = vals_reshape_dict[drought_year]
            gs_vals = drought_year_vals[gs_mon_index]
            gs_vals_mean = np.nanmean(gs_vals)
            gs_vals_list.append(gs_vals_mean)
        df[var_name+'_GS'] = gs_vals_list
        return df

    def add_GS_Precipitation(self,df):
        spatial_dict,var_name = Load_Data().Precipitation_anomaly_detrend()
        year_list = global_year_range_list
        gs_mon_index = np.array(global_gs,dtype=int) - 1
        gs_vals_list = []
        for i,row in tqdm(df.iterrows(),total=len(df),desc='add_GS_Precipitation'):
            pix = row['pix']
            drought_year = row['drought_year']
            vals = spatial_dict[pix]
            vals_reshape = np.array(vals).reshape(-1,12)
            vals_reshape_dict = dict(zip(year_list,vals_reshape))
            drought_year_vals = vals_reshape_dict[drought_year]
            gs_vals = drought_year_vals[gs_mon_index]
            gs_vals_mean = np.nanmean(gs_vals)
            gs_vals_list.append(gs_vals_mean)
        df[var_name+'_GS'] = gs_vals_list
        return df

    def add_GS_SPI(self,df):
        spatial_dict,var_name = Load_Data().SPI()
        year_list = global_year_range_list
        gs_mon_index = np.array(global_gs,dtype=int) - 1
        gs_vals_list = []
        for i,row in tqdm(df.iterrows(),total=len(df),desc='add_GS_SPI'):
            pix = row['pix']
            drought_year = row['drought_year']
            vals = spatial_dict[pix]
            vals_reshape = np.array(vals).reshape(-1,12)
            vals_reshape_dict = dict(zip(year_list,vals_reshape))
            drought_year_vals = vals_reshape_dict[drought_year]
            gs_vals = drought_year_vals[gs_mon_index]
            gs_vals_mean = np.nanmean(gs_vals)
            gs_vals_list.append(gs_vals_mean)
        df[var_name+'_GS'] = gs_vals_list
        return df

    def add_GS_NDVI(self,df):
        spatial_dict,var_name = Load_Data().NDVI_anomaly_detrend()
        year_list = global_year_range_list
        gs_mon_index = np.array(global_gs,dtype=int) - 1
        gs_vals_list = []
        for i,row in tqdm(df.iterrows(),total=len(df),desc='add_GS_NDVI'):
            pix = row['pix']
            drought_year = row['drought_year']
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

    def random_forest_train(self):
        fdir = join(self.this_class_arr,'seasonal_delta')
        outdir = join(self.this_class_arr,'random_forest')
        T.mk_dir(outdir,force=True)
        variables_dict = {
            'spring':['SOS','Precipitation-anomaly_spring','Temperature-anomaly_spring','SPI_spring',],
            'summer':['SOS','Precipitation-anomaly_summer','Temperature-anomaly_summer','SPI_summer'],
            'autumn':['EOS','Precipitation-anomaly_autumn','Temperature-anomaly_autumn','SPI_autumn',]
        }
        for season in variables_dict:
            dff = join(fdir,f'{season}.df')
            df = T.load_df(dff)
            x_variable_list = variables_dict[season]
            y_variable = 'delta'
            df = df.dropna(subset=x_variable_list+[y_variable])
            X = df[x_variable_list]
            Y = df[y_variable]
            clf, importances_dic, mse, r_model, score, Y_test, y_pred = self._random_forest_train(X, Y, x_variable_list)
            print('-----------------------------------')
            print(importances_dic)
            print('r_model',r_model)
            print('score',score)
            # exit()



    def _random_forest_train(self, X, Y, variable_list):
        '''
        :param X: a dataframe of x variables
        :param Y: a dataframe of y variable
        :param variable_list: a list of x variables
        :return: details of the random forest model and the importance of each variable
        '''
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) # split the data into training and testing
        clf = RandomForestRegressor(n_estimators=100, n_jobs=7) # build a random forest model
        clf.fit(X_train, Y_train) # train the model
        result = permutation_importance(clf, X_train, Y_train, scoring=None,
                                        n_repeats=50, random_state=1,
                                        n_jobs=7) # calculate the importance of each variable using permutation importance
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


    def plot_rf_result(self):
        fdir = join(self.this_class_arr,'random_forest')
        outdir = join(self.this_class_png,'random_forest')
        T.mk_dir(outdir,force=True)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            result_dict = T.load_npy(fpath)
            importances_dic = result_dict['importances_dic']
            r_model = result_dict['r_model']
            score = result_dict['score']
            title = f'{f}\nR^2={score}, r={r_model}'
            x = importances_dic.keys()
            y = [importances_dic[key] for key in x]
            plt.figure(figsize=(10,5))
            plt.bar(x,y)
            plt.title(title)
            outf = join(outdir,f'{f}.pdf')
            plt.savefig(outf,dpi=300)
            plt.close()
        T.open_path_and_file(outdir)


class Partial_Dependence_Plots:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Partial_Dependence_Plots', result_root_this_script, mode=2)
        self.dff = Random_Forests().dff
        pass

    def run(self):
        self.run_partial_dependence_plots()
        # self.plot_run_partial_dependence_plots()
        pass

    def run_partial_dependence_plots(self):
        dff = self.dff
        df = T.load_df(dff)
        outdir = join(self.this_class_arr,'partial_dependence_plots')
        T.mk_dir(outdir,force=True)
        variable_list_dict = {
            'early': ['SOS', 'Precipitation-anomaly_early', 'Temperature-anomaly_early', 'SPI_early', ],
            'mid': ['SOS', 'Precipitation-anomaly_early', 'Temperature-anomaly_early', 'Precipitation-anomaly_mid',
                    'Temperature-anomaly_mid'],
            'late': ['EOS', 'Precipitation-anomaly_late', 'Temperature-anomaly_late', 'SPI_late', ]
        }
        print(df.columns.tolist())

        drought_type_list = global_drought_type_list
        ELI_class_list = global_ELI_class_list

        for drought_type in drought_type_list:
            for ELI_class in ELI_class_list:
                for season in variable_list_dict:
                    print('train', drought_type, ELI_class, season)
                    df_drt = df[df['drought_type'] == drought_type]
                    df_ELI = df_drt[df_drt['ELI_class'] == ELI_class]
                    df_season = df_ELI[df_ELI['season'] == season]
                    result_dic = self.partial_dependence_plots(df_season, variable_list_dict[season], 'Y')
                    outf = join(outdir, f'{drought_type}_{ELI_class}_{season}.npy')
                    T.save_npy(result_dic, outf)


    def plot_run_partial_dependence_plots(self):
        fdir = join(self.this_class_arr,'partial_dependence_plots')
        outdir = join(self.this_class_png,'partial_dependence_plots')
        T.mk_dir(outdir,force=True)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            result_dict = T.load_npy(fpath)
            plt.figure(figsize=(5, 7))
            flag = 1
            for key in result_dict:
                result_dict_i = result_dict[key]
                x = result_dict_i['x']
                y = result_dict_i['y']
                y_std = result_dict_i['y_std']
                plt.subplot(3,2,flag)
                flag += 1
                plt.plot(x,y,label=key)
                plt.fill_between(x,y-y_std,y+y_std,alpha=0.5)
                plt.legend()
                plt.ylim(-1.5,1)
            plt.suptitle(f)
            plt.tight_layout()
            outf = join(outdir,f'{f}.pdf')
            plt.savefig(outf,dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=1, test_size=0.2) # split the data into training and testing
        rf = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=4) # build a random forest model
        rf.fit(X_train, y_train) # train the model
        r2 = rf.score(X_test,y_test)
        return rf,r2

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


class Partial_Dependence_Plots_Delta:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Partial_Dependence_Plots_Delta', result_root_this_script, mode=2)

        self.variables_dict = {
            'spring': [
                'SOS_delta',
                # 'Precipitation-anomaly_delta',
                # 'Temperature-anomaly_detrend_delta',
                'SPI_delta',
                # 'Radiation-anomaly_detrend_delta',
                'FAPAR-anomaly_detrend_delta',
                'Temperature_anomaly_GS_delta',

                # 'Precipitation-anomaly_detrend_vs_NDVI-anomaly_detrend_spring',
                # 'Temperature-anomaly_detrend_vs_NDVI-anomaly_detrend_spring',
                # 'SPI_vs_NDVI-anomaly_detrend_spring',
                # 'early_start_vs_NDVI-anomaly_detrend_spring'
                       ],
            'summer': [
                'SOS_delta',
                # 'Precipitation-anomaly_delta',
                # 'Temperature-anomaly_detrend_delta',
                'SPI_delta',
                # 'Radiation-anomaly_detrend_delta',
                'FAPAR-anomaly_detrend_delta',
                'Temperature_anomaly_GS_delta',

                # 'Precipitation-anomaly_detrend_vs_NDVI-anomaly_detrend_summer',
                # 'Temperature-anomaly_detrend_vs_NDVI-anomaly_detrend_summer',
                # 'SPI_vs_NDVI-anomaly_detrend_summer',
                # 'early_start_vs_NDVI-anomaly_detrend_summer'
                          ],
            'autumn': [
                'SOS_delta','EOS_delta',
                # 'Precipitation-anomaly_delta',
                # 'Temperature-anomaly_detrend_delta',
                'SPI_delta',
                # 'Radiation-anomaly_detrend_delta',
                'FAPAR-anomaly_detrend_delta',
                'Temperature_anomaly_GS_delta',

                # 'Precipitation-anomaly_detrend_vs_NDVI-anomaly_detrend_autumn',
                # 'Temperature-anomaly_detrend_vs_NDVI-anomaly_detrend_autumn',
                # 'SPI_vs_NDVI-anomaly_detrend_autumn',
                # 'late_end_vs_NDVI-anomaly_detrend_autumn'
                          ]
        }
        pass

    def run(self):
        # self.run_importance()
        # self.plot_importance()
        self.print_model_performance()
        # self.run_partial_dependence_plots()
        # self.plot_run_partial_dependence_plots()
        pass

    def run_partial_dependence_plots(self):
        # fdir = join(Random_Forests_delta().this_class_arr, 'seasonal_delta')
        fdir = join(Random_Forests_delta().this_class_arr, 'seasonal_delta_merge')
        outdir = join(self.this_class_arr, 'partial_dependence_plots')
        T.mk_dir(outdir, force=True)
        variables_dict = self.variables_dict
        ELI_class_list = global_ELI_class_list
        for season in variables_dict:
            dff = join(fdir, f'{season}.df')
            df = T.load_df(dff)
            df = self.clean_df(df)

            df = self.__add_sensitivity(df,season)
            # T.print_head_n(df)
            # exit()
            print('training',season)
            print('-'*50)
            for ELI in ELI_class_list:
                df_ELI = df[df['ELI_class']==ELI]

                result_dic = self.partial_dependence_plots(df_ELI, variables_dict[season], 'NDVI_delta')
                outf = join(outdir, f'{season}_{ELI}.npy')
                T.save_npy(result_dic, outf)


    def plot_run_partial_dependence_plots(self):
        fdir = join(self.this_class_arr,'partial_dependence_plots')
        outdir = join(self.this_class_png,'partial_dependence_plots')
        T.mk_dir(outdir,force=True)
        season_list = global_drought_season_list
        ELI_class_list = global_ELI_class_list

        for season in season_list:
            plt.figure(figsize=(7, 7))
            for ELI in ELI_class_list:
                fpath = join(fdir,f'{season}_{ELI}.npy')

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
                    plt.ylim(-.5,.5)
                    plt.xlabel(key.replace('_vs_NDVI-anomaly_detrend_','\nsensitivity\n'))
                plt.legend()
            plt.suptitle(season)

            plt.tight_layout()
            outf = join(outdir,f'{season}.pdf')
            # plt.savefig(outf,dpi=300)
            plt.show()
        T.open_path_and_file(outdir)

    def run_importance(self):
        fdir = join(Random_Forests_delta().this_class_arr, 'seasonal_delta_merge')
        outdir = join(self.this_class_arr, 'importance')
        T.mk_dir(outdir, force=True)
        ELI_class_list = global_ELI_class_list
        variables_dict = self.variables_dict
        for season in variables_dict:
            dff = join(fdir, f'{season}.df')
            df = T.load_df(dff)
            df = self.clean_df(df)
            # T.print_head_n(df)
            # exit()
            print('training', season)
            print('-' * 50)
            for ELI in ELI_class_list:
                df_ELI = df[df['ELI_class'] == ELI]
                df_ELI = self.__add_sensitivity(df_ELI,season)
                # print(df_ELI.columns.tolist())

                df_ELI = df_ELI.dropna(subset=variables_dict[season] + ['NDVI_delta'])

                X = df_ELI[variables_dict[season]]
                Y = df_ELI['NDVI_delta']
                clf, importances_dic, mse, r_model, score, Y_test, y_pred = (
                    self._random_forest_train(X, Y, variables_dict[season]))
                result_dic = {
                    'importances_dic':importances_dic,
                    'r_model':r_model,
                    'score':score,
                }
                outf = join(outdir, f'{season}_{ELI}.npy')
                T.save_npy(result_dic, outf)

    def plot_importance(self):
        fdir = join(self.this_class_arr, 'importance')
        outdir = join(self.this_class_png, 'importance')
        T.mk_dir(outdir, force=True)
        ELI_class_list = global_ELI_class_list
        season_list = global_drought_season_list

        for season in season_list:
            x_list = []
            y_list = []
            for ELI_class in ELI_class_list:
                f = f'{season}_{ELI_class}.npy'
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
            y_list = np.array(y_list)
            # y_list_mean = np.nanmean(y_list,axis=0)
            # sort
            # y_list_mean_sort = np.argsort(y_list_mean)
            # x_list_sort = [x_list[0][i] for i in y_list_mean_sort]
            flag = 0
            for i in range(len(x_list)):
                plt.figure(figsize=(5 * centimeter_factor, 10 * centimeter_factor))

                plt.scatter(y_list[i],x_list[i],s=100,marker='o',c='none',linewidths=2,
                            edgecolors=global_ELI_class_color_dict[ELI_class_list[flag]],
                            label=ELI_class_list[flag])
                flag += 1
                # plt.xlim(0, 1)


                plt.title(f'{season}_{ELI_class_list[i]}')
                # plt.legend()
                outf = join(outdir, f'{season}_{ELI_class_list[i]}.pdf')
                plt.tight_layout()

                plt.savefig(outf, dpi=300)
                plt.close()
            # plt.show()

        T.open_path_and_file(outdir)

    def print_model_performance(self):
        fdir = join(self.this_class_arr, 'partial_dependence_plots')
        season_list = global_drought_season_list
        ELI_class_list = global_ELI_class_list

        for season in season_list:
            for ELI in ELI_class_list:
                fpath = join(fdir, f'{season}_{ELI}.npy')

                result_dict = T.load_npy(fpath)
                flag = 1
                for key in result_dict:
                    result_dict_i = result_dict[key]
                    x = result_dict_i['x']
                    y = result_dict_i['y']
                    y_std = result_dict_i['y_std']
                    r2 = result_dict_i['r2']
                print(fpath,r2)
        exit()

        pass

    def clean_df(self,df):
        delta_threshold_dict = Random_Forests_delta().delta_threshold()
        for key in delta_threshold_dict:
            left,right = delta_threshold_dict[key]
            print(key,(left,right))
            df = df[df[key]>=left]
            df = df[df[key]<=right]

        return df

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

    def _random_forest_train(self, X, Y, variable_list):
        '''
        :param X: a dataframe of x variables
        :param Y: a dataframe of y variable
        :param variable_list: a list of x variables
        :return: details of the random forest model and the importance of each variable
        '''
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) # split the data into training and testing
        clf = RandomForestRegressor(n_estimators=100, n_jobs=7) # build a random forest model
        clf.fit(X_train, Y_train) # train the model
        result = permutation_importance(clf, X_train, Y_train, scoring=None,
                                        n_repeats=50, random_state=1,
                                        n_jobs=7) # calculate the importance of each variable using permutation importance
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

    def __add_sensitivity(self,df,season):
        import analysis
        fdir = join(analysis.Long_term_correlation().this_class_tif,
                    'seasonal_correlation')
        print('adding', season)
        for folder in T.listdir(fdir):
            fpath = join(fdir,folder,f'{folder}_{season}.tif')
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            df = T.add_spatial_dic_to_df(df,spatial_dict,f'{folder}_{season}')
        return df


    def __train_model(self,X,y):
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=1, test_size=0.2) # split the data into training and testing
        rf = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=4) # build a random forest model
        rf.fit(X_train, y_train) # train the model
        r2 = rf.score(X_test,y_test)
        return rf,r2

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

class Single_Variable_Analysis:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Single_Variable_Analysis', result_root_this_script, mode=2)
        self.dff = Random_Forests_delta().dff
        pass

    def run(self):
        # self.run_corr()
        # self.plot_corr()
        self.bin_plot()
        # self.plot_delta()
        pass

    def run_corr(self):
        dff = self.dff
        df = T.load_df(dff)
        outdir = join(self.this_class_arr, 'correlation')
        T.mk_dir(outdir, force=True)
        variable_list_dict = {
            'early': ['SOS', 'Precipitation-anomaly_early', 'Temperature-anomaly_early', 'SPI_early', ],
            'mid': ['SOS', 'Precipitation-anomaly_early', 'Temperature-anomaly_early', 'Precipitation-anomaly_mid',
                    'Temperature-anomaly_mid'],
            'late': ['EOS', 'Precipitation-anomaly_late', 'Temperature-anomaly_late', 'SPI_late', ]
        }
        print(df.columns.tolist())

        drought_type_list = global_drought_type_list
        ELI_class_list = global_ELI_class_list

        for drought_type in drought_type_list:
            for ELI_class in ELI_class_list:
                for season in variable_list_dict:
                    print('train', drought_type, ELI_class, season)
                    df_drt = df[df['drought_type'] == drought_type]
                    df_ELI = df_drt[df_drt['ELI_class'] == ELI_class]
                    df_season = df_ELI[df_ELI['season'] == season]
                    variable_list = variable_list_dict[season]
                    result_dic = {}
                    for var_i in variable_list:
                        x = df_season[var_i].values
                        y = df_season['Y'].values
                        r, p = T.nan_correlation(x, y)
                        result_dic[var_i] = {'r': r, 'p': p}
                    outf = join(outdir, f'{drought_type}_{ELI_class}_{season}.npy')
                    T.save_npy(result_dic, outf)

        pass

    def plot_corr(self):
        fdir = join(self.this_class_arr, 'correlation')
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            result_dic = T.load_npy(fpath)
            # print(result_dic)
            # exit()
            plt.figure(figsize=(10, 5))
            x_list = []
            y_list = []
            for x_i in result_dic:
                x_list.append(x_i)
                y = result_dic[x_i]['r']
                y_list.append(y)
            plt.bar(x_list, y_list)
        plt.show()


    def bin_plot(self):
        # dff_fdir = join(Random_Forests_delta().this_class_arr, 'seasonal_delta_merge')
        dff_fdir = join(Random_Forests_delta().this_class_arr, 'seasonal_delta')
        # var_in = 'Radiation-anomaly_detrend_delta'
        # var_in = Load_Data().Temperature_anomaly_detrend()
        # var_in = 'Temperature-anomaly_juping_detrend_delta'
        var_in = 'Temperature_anomaly_GS_delta'
        # var_in = 'Temperature-anomaly_detrend_delta'
        outdir = join(self.this_class_png, 'bin_plot')
        T.mk_dir(outdir, force=True)
        bin_dict = {
            'SOS': np.linspace(-25, 25, 21),
            'EOS': np.linspace(-25, 25, 21),
        }

        drought_type_list = global_drought_type_list
        ELI_class_list = global_ELI_class_list
        flag = 1
        plt.figure(figsize=(5, 7))

        for season in global_drought_season_list:
            dff_season = join(dff_fdir,f'{season}.df')
            df = T.load_df(dff_season)
            print(df.columns)
            for ELI_class in ELI_class_list:
                df_ELI = df[df['ELI_class'] == ELI_class]
                df_season = df_ELI[df_ELI['drought_season'] == season]
                # variable_list = variable_list_dict[season]
                variable_list = [var_in]
                result_dic = {}
                plt.subplot(3, 2, flag)
                for var_i in variable_list:
                    # bins = np.linspace(-3, 3, 21)
                    bins = np.linspace(0, 2, 41)
                    vals = df_season[var_i].tolist()
                    # plt.hist(vals,bins=100)
                    # plt.show()

                    if var_i in bin_dict:
                        bins = bin_dict[var_i]
                    df_group,bins_list_str = T.df_bin(df_season, var_i, bins)
                    x_list = []
                    y_list = []
                    err_list = []
                    for name,df_group_i in df_group:
                        x = name[0].left
                        vals = df_group_i['NDVI_delta'].tolist()
                        if len(vals) == 0:
                            continue
                        mean = np.nanmean(vals)
                        x_list.append(x)
                        y_list.append(mean)
                        err,_,_ = T.uncertainty_err(vals)
                        err_list.append(err)

                    flag += 1
                    plt.plot(x_list, y_list, label=var_i)
                    plt.fill_between(x_list, np.array(y_list) - np.array(err_list),
                                        np.array(y_list) + np.array(err_list), alpha=0.2)
                    plt.title(f'{ELI_class}_{season}')
                    # plt.legend()
                    # plt.ylim(-1.5, 1.5)
                # plt.suptitle(f'{ELI_class}_{season}')
        plt.suptitle(f'{var_in}')
        plt.tight_layout()

        outf = join(outdir, f'{var_in}.pdf')
        plt.savefig(outf, dpi=300)
        plt.close()
        T.open_path_and_file(outdir)

    def plot_delta(self):
        dff_fdir = join(Random_Forests_delta().this_class_arr, 'seasonal_delta')
        outdir = join(self.this_class_tif, 'delta')
        T.mk_dir(outdir, force=True)
        # var_in = 'Radiation-anomaly_detrend_delta'
        # var_in = Load_Data().Temperature_anomaly_detrend()
        # var_in = 'Temperature-anomaly_delta'
        var_in = 'NDVI_delta'

        drought_type_list = global_drought_type_list
        ELI_class_list = global_ELI_class_list
        flag = 1
        # plt.figure(figsize=(5, 7))

        for season in global_drought_season_list:
            dff_season = join(dff_fdir, f'{season}.df')
            df = T.load_df(dff_season)
            spatial_dict = T.df_to_spatial_dic(df, var_in)
            outf = join(outdir,f'{var_in}_{season}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict,outf)
        T.open_path_and_file(outdir)

class Phenology_Statistic:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Phenology_Statistic', result_root_this_script, mode=2)
        self.dff = Random_Forests().dff
        pass

    def run(self):
        # df = self.__gen_df_init()
        # self.tif_SOS_EOS(df)

        # self.plot_tif_SOS_EOS()
        # self.plot_pdf_SOS_EOS()
        self.bar_SOS_EOS()
        # self.bar_SOS_EOS_all()

        pass

    def tif_SOS_EOS(self,df):
        outdir = join(self.this_class_tif,'SOS_EOS')
        T.mk_dir(outdir,force=True)
        var_list = ['SOS','EOS']
        drought_type_list = global_drought_type_list
        for var_i in var_list:
            for drt in drought_type_list:
                df_drt = df[df['drought_type']==drt]
                df_group_dict = T.df_groupby(df_drt,'pix')
                spatial_dict = {}
                for pix in tqdm(df_group_dict,desc=f'{var_i}_{drt}'):
                    df_i = df_group_dict[pix]
                    vals = df_i[var_i].tolist()
                    mean = np.nanmean(vals)
                    spatial_dict[pix] = mean
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
                outf = join(outdir,f'{var_i}_{drt}.tif')
                DIC_and_TIF().arr_to_tif(arr,outf)
        T.open_path_and_file(outdir)


    def plot_tif_SOS_EOS(self):
        fdir = join(self.this_class_tif,'SOS_EOS')
        outdir = join(self.this_class_png,'SOS_EOS')
        # color_list = ['purple', '#FFFFCC', 'g'][::-1]
        color_list = ['purple', '#FFFFCC', 'g'][::-1]
        color_list_r = ['purple', '#FFFFCC', 'g']
        cmap = T.cmap_blend(color_list)
        cmap_r = T.cmap_blend(color_list_r)
        plt.register_cmap(name='mycmap', cmap=cmap)
        plt.register_cmap(name='mycmap_r', cmap=cmap_r)
        T.mk_dir(outdir,force=True)
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            if 'SOS' in f:
                cmap = 'mycmap'
            else:
                cmap = 'mycmap_r'
            Plot().plot_ortho(fpath,vmin=-10,vmax=10,cmap=cmap)
            plt.title(f.replace('.tif',''))
            outf = join(outdir,f.replace('.tif','.png'))
            plt.savefig(outf,dpi=300)
            plt.close()
            # plt.show()
        T.open_path_and_file(outdir)
        pass

    def plot_pdf_SOS_EOS(self):
        fdir = join(self.this_class_tif,'SOS_EOS')
        outdir = join(self.this_class_png,'pdf_SOS_EOS')
        T.mk_dir(outdir,force=True)
        all_dict = {}
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            key = f.replace('.tif','')
            all_dict[key] = spatial_dict
        df = T.spatial_dics_to_df(all_dict)
        df = Dataframe_func(df).df
        T.print_head_n(df)

        var_list = ['SOS','EOS']
        drought_type_list = global_drought_type_list
        ELI_class_list = global_ELI_class_list
        for var_i in var_list:
            for ELI in ELI_class_list:
                df_eli = df[df['ELI_class']==ELI]
                plt.figure(figsize=(12*centimeter_factor,6*centimeter_factor))
                for drt in drought_type_list:
                    key = f'{var_i}_{drt}'
                    values = df_eli[key].tolist()
                    x,y = Plot().plot_hist_smooth(values,alpha=0,bins=100,range=(-30,30))
                    plt.plot(x,y,label=f'{drt}')
                plt.legend()
                plt.title(f'{var_i}_{ELI}')
                plt.tight_layout()
                outf = join(outdir,f'{var_i}_{ELI}.pdf')
                plt.savefig(outf,dpi=300)
                plt.close()
                # plt.show()
        T.open_path_and_file(outdir)
        pass

    def bar_SOS_EOS(self):
        fdir = join(self.this_class_tif,'SOS_EOS')
        outdir = join(self.this_class_png,'bar_SOS_EOS')
        T.mk_dir(outdir,force=True)
        all_dict = {}
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            key = f.replace('.tif','')
            all_dict[key] = spatial_dict
        df = T.spatial_dics_to_df(all_dict)
        df = Dataframe_func(df).df
        T.print_head_n(df)

        var_list = ['SOS','EOS']
        drought_type_list = global_drought_type_list
        ELI_class_list = global_ELI_class_list
        for var_i in var_list:
            x_list = []
            y_list = []
            plt.figure(figsize=(12 * centimeter_factor, 6 * centimeter_factor))
            for ELI in ELI_class_list:
                df_eli = df[df['ELI_class']==ELI]
                for drt in drought_type_list:
                    key = f'{var_i}_{drt}'
                    values = df_eli[key].tolist()
                    values = np.array(values)
                    values = T.remove_np_nan(values)
                    values_greater_than_0 = values[values>0]
                    values_less_than_0 = values[values<0]
                    ratio_greater_than_0 = len(values_greater_than_0)/len(values)
                    ratio_less_than_0 = len(values_less_than_0)/len(values)
                    x_list.append(f'{ELI}_{drt}')
                    x_list.append(f'{ELI}_{drt}')
                    y_list.append(ratio_greater_than_0)
                    y_list.append(-ratio_less_than_0)
                    plt.text(ratio_greater_than_0,f'{ELI}_{drt}',f'{ratio_greater_than_0:.2f}',ha='left',va='center')
                    plt.text(-ratio_less_than_0, f'{ELI}_{drt}', f'{ratio_less_than_0:.2f}', ha='right', va='center')
            plt.barh(x_list,y_list)
            plt.title(f'{var_i}')
            plt.tight_layout()
            plt.xlim(-1,1)
            plt.vlines(0,-1,4,color='r',zorder=100)
            outf = join(outdir,f'{var_i}_regions.pdf')
            plt.savefig(outf,dpi=300)
            plt.close()
        # plt.show()
        T.open_path_and_file(outdir)
        pass

    def bar_SOS_EOS_all(self):
        fdir = join(self.this_class_tif, 'SOS_EOS')
        outdir = join(self.this_class_png, 'bar_SOS_EOS')
        T.mk_dir(outdir, force=True)
        all_dict = {}
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir, f)
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            key = f.replace('.tif', '')
            all_dict[key] = spatial_dict
        df = T.spatial_dics_to_df(all_dict)
        df = Dataframe_func(df).df
        T.print_head_n(df)

        var_list = ['SOS', 'EOS']
        drought_type_list = global_drought_type_list
        ELI = 'all'
        for var_i in var_list:
            x_list = []
            y_list = []
            plt.figure(figsize=(12 * centimeter_factor, 6 * centimeter_factor))
            df_eli = df
            for drt in drought_type_list:
                key = f'{var_i}_{drt}'
                values = df_eli[key].tolist()
                values = np.array(values)
                values = T.remove_np_nan(values)
                values_greater_than_0 = values[values > 0]
                values_less_than_0 = values[values < 0]
                ratio_greater_than_0 = len(values_greater_than_0) / len(values)
                ratio_less_than_0 = len(values_less_than_0) / len(values)
                x_list.append(f'{ELI}_{drt}')
                x_list.append(f'{ELI}_{drt}')
                y_list.append(ratio_greater_than_0)
                y_list.append(-ratio_less_than_0)
                plt.text(ratio_greater_than_0,f'{ELI}_{drt}',f'{ratio_greater_than_0:.2f}',ha='left',va='center')
                plt.text(-ratio_less_than_0, f'{ELI}_{drt}', f'{ratio_less_than_0:.2f}', ha='right', va='center')
            plt.barh(x_list, y_list)
            plt.title(f'{var_i}')
            plt.tight_layout()
            plt.xlim(-1,1)
            plt.vlines(0,-1,2,color='r',zorder=100)
            outf = join(outdir,f'{var_i}.pdf')
            plt.savefig(outf,dpi=300)
            plt.close()
        # plt.show()
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

class Optimal_temperature_statistic:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Optimal_temperature_statistic', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'Dataframe.df')
        pass

    def run(self):
        # self.copy_df()
        self.add_optimal_temperature()
        # self.add_optimal_temperature_monthly()
        # self.add_NDVI_origin_process()
        # self.foo()
        # self.delta_optimal_temp()
        # self.delta_vs_rt()
        # self.delta_vs_NDVI()
        # self.delta_vs_NDVI_one_pixel()
        # self.timing_trajectory_Tair()
        # self.T_NDVI_relation()
        self.T_NDVI_relation_CSIF()
        # self.T_NDVI_relation_hist()
        pass

    def copy_df(self):
        if isfile(self.dff):
            print('already exists: ', self.dff)
            print('press enter to overwrite')
            pause()
            pause()
            pause()
        dff = Drought_timing().dff
        df = T.load_df(dff)
        T.save_df(df,self.dff)
        T.df_to_excel(df, self.dff)

    def add_optimal_temperature(self):
        import analysis
        df = T.load_df(self.dff)
        Topt_f = join(analysis.Optimal_temperature().this_class_tif,'optimal_temperature/LT_Baseline_NT_origin_step_0.5_celsius_resample.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(Topt_f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'Topt')
        T.print_head_n(df)
        return df
        pass

    def add_optimal_temperature_monthly(self):
        import analysis
        df = T.load_df(self.dff)
        Topt_dir = join(analysis.Optimal_temperature_monthly().this_class_tif,r'optimal_temperature_mean')
        spatial_dict_all = {}
        for mon in range(1,13):
            fpath = join(Topt_dir,f'optimal_temperature_{mon:02d}.tif')
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dict_all[mon] = spatial_dict

        monthly_opt_T_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_mon = row['drought_mon']
            monthly_opt_T = spatial_dict_all[drought_mon][pix]
            monthly_opt_T_list.append(monthly_opt_T)
        df['monthly_opt_T'] = monthly_opt_T_list
        T.print_head_n(df)
        T.save_df(df,self.dff)
        T.df_to_excel(df, self.dff)



    def add_NDVI_origin_process(self):
        df = T.load_df(self.dff)
        NDVI_spatial_dict,_ = Load_Data().NDVI_origin()
        year_list = global_year_range_list
        gs = global_gs
        NDVI_list_all = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            NDVI = NDVI_spatial_dict[pix]
            NDVI = np.array(NDVI,dtype=float)
            NDVI[NDVI>10000] = np.nan
            NDVI[NDVI<0] = np.nan
            NDVI_gs = T.monthly_vals_to_annual_val(NDVI,gs,method='array')
            NDVI_gs_dict = T.dict_zip(year_list,NDVI_gs)
            NDVI_list = []
            year_list_i = []
            for y in range(-1,5):
                y_i = drought_year+y
                if y_i in NDVI_gs_dict:
                    NDVI_list.append(NDVI_gs_dict[drought_year+y])
                else:
                    NDVI_list.append([np.nan]*len(gs))
                year_list_i.append(y_i)
            NDVI_list = np.array(NDVI_list)
            NDVI_list = NDVI_list.flatten()
            NDVI_list_all.append(NDVI_list)
        df['NDVI_origin_progress'] = NDVI_list_all
        T.save_df(df,self.dff)
        T.df_to_excel(df, self.dff)

    def foo(self):
        df = T.load_df(self.dff)
        T.print_head_n(df)
        # df = df[df['drought_type']=='hot-drought']
        df = df[df['drought_type']=='normal-drought']
        eli_class = global_ELI_class # ['Energy-Limited', 'Water-Limited']
        for eli in eli_class:
            df_eli = df[df['ELI_class'] == eli]
            temperature_list = []
            opt_list = []
            for i,row in tqdm(df_eli.iterrows(),total=len(df_eli)):
                Topt = row['optimal_temp']
                if np.isnan(Topt):
                    continue
                Temperature_origin_progress = row['Temperature-origin_progress']
                opt_list.append(Topt)
                temperature_list.append(Temperature_origin_progress)
            temperature_list = np.array(temperature_list)
            temperature_list_mean = np.nanmean(temperature_list, axis=0)
            temperature_list_std = np.nanstd(temperature_list, axis=0)
            opt_list = np.array(opt_list)
            opt_list_mean = np.nanmean(opt_list)
            # pix = row['pix']
            # print(eli,pix,Topt)
            plt.plot(temperature_list_mean)
            plt.plot(range(len(temperature_list_mean)),[opt_list_mean]*len(temperature_list_mean))
            # print(len(Temperature_origin_progress))
            plt.show()

    def delta_optimal_temp(self):
        outdir = join(self.this_class_tif, 'delta_optimal_temp_monthly')
        T.mk_dir(outdir,force=1)
        df = T.load_df(self.dff)
        df = df[df['drought_season']!='other']
        drought_mon = df['drought_mon'].tolist()
        # plt.hist(drought_mon,bins=6)
        # plt.show()

        delta_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            # Topt = row['optimal_temp']
            Topt = row['monthly_opt_T']
            if np.isnan(Topt):
                delta_list.append(np.nan)
                continue
            Temperature_origin_progress = row['Temperature-origin_progress']
            Temperature_origin_progress = np.array(Temperature_origin_progress)
            Temperature_origin_progress_reshape = np.reshape(Temperature_origin_progress, (-1, 6))
            Temperature_origin_drought_year = Temperature_origin_progress_reshape[1]
            # opt_list.append(Topt)
            drought_mon = row['drought_mon']
            indx = drought_mon - global_gs[0]
            # print(indx)
            # print(drought_mon)
            # exit()
            drought_mon_temp = Temperature_origin_drought_year[indx]
            delta = drought_mon_temp - Topt
            delta_list.append(delta)
        df['delta_temp'] = delta_list
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)

        drt_list = ['hot-drought', 'normal-drought']
        for drt in drt_list:
            df_drt = df[df['drought_type'] == drt]
            df_pix_group = T.df_groupby(df_drt, 'pix')
            spatial_dict = {}
            for pix in df_pix_group:
                df_pix = df_pix_group[pix]
                delta_temp = df_pix['delta_temp']
                delta_temp_mean = np.nanmean(delta_temp)
                spatial_dict[pix] = delta_temp_mean
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
            outf = join(outdir, drt + '.tif')
            DIC_and_TIF().arr_to_tif(arr, outf)

    def delta_vs_rt(self): # not working
        df = T.load_df(self.dff)
        T.print_head_n(df)
        drt_list = ['hot-drought', 'normal-drought']
        eli_list = ['Energy-Limited', 'Water-Limited']
        for eli in eli_list:
            df_eli = df[df['ELI_class'] == eli]
            for drt in drt_list:
                df_drt = df_eli[df_eli['drought_type'] == drt]
                df_drt = df_drt.dropna(subset=['rt', 'delta_temp'], how='any')
                # rt = df_drt['rt'].tolist()
                # delta = df_drt['delta_temp'].tolist()
                # delta = np.array(delta)
                max_NDVI_list = []
                delta_list = []
                for i,row in df_drt.iterrows():
                    NDVI_progress = row['NDVI_origin_progress']
                    NDVI_progress = np.array(NDVI_progress)
                    NDVI_progress = np.reshape(NDVI_progress, (-1, 6))
                    NDVI_progress = NDVI_progress[1]
                    max_NDVI = np.nanmax(NDVI_progress)
                    max_NDVI_list.append(max_NDVI)

                    Temp_process = row['Temperature-origin_progress']
                    Temp_process = np.array(Temp_process)
                    Temp_process = np.reshape(Temp_process, (-1, 6))
                    Temp_process = Temp_process[1]
                    max_Temp = np.nanmax(Temp_process)

                    T_opt = row['optimal_temp']

                    delta = max_Temp - T_opt
                    delta_list.append(delta)

                # print(len(rt), len(delta))
                KDE_plot().plot_scatter(delta_list, max_NDVI_list,s=4)
                plt.title(f'{eli}-{drt}')
                plt.show()

    def delta_vs_NDVI(self):
        df = T.load_df(self.dff)
        T.print_head_n(df)
        eli_list = ['Energy-Limited', 'Water-Limited']
        drt_list = ['hot-drought', 'normal-drought']
        for eli in eli_list:
            for drt in drt_list:
                df_eli = df[df['ELI_class'] == eli]
                df_drt = df_eli[df_eli['drought_type'] == drt]
                x_list = []
                y_list = []
                opt_list = []
                # for i, row in tqdm(df_drt.iterrows(), total=len(df_drt)):
                for i, row in tqdm(df_drt.iterrows(), total=len(df_drt)):
                    NDVI_progress = row['NDVI_origin_progress']
                    NDVI_progress = np.array(NDVI_progress)
                    NDVI_progress = np.reshape(NDVI_progress, (-1, 6))
                    NDVI_progress = NDVI_progress[1]
                    # max_NDVI = np.nanmax(NDVI_progress)
                    max_NDVI = np.nanmean(NDVI_progress)

                    Temp_process = row['Temperature-origin_progress']
                    Temp_process = np.array(Temp_process)
                    Temp_process = np.reshape(Temp_process, (-1, 6))
                    Temp_process = Temp_process[1]
                    max_Temp = np.nanmax(Temp_process)
                    # max_Temp = np.nanmean(Temp_process)
                    x_list.append(max_Temp)
                    y_list.append(max_NDVI)
                    T_opt = row['optimal_temp']
                    opt_list.append(T_opt)
                opt_temp_mean = np.nanmean(opt_list)
                # plt.figure()
                # plt.scatter(x_list, y_list)
                KDE_plot().plot_scatter(x_list, y_list,s=4,max_n=50000)
                plt.vlines(opt_temp_mean, 0, 10000, 'r')
                plt.title(f'{eli}-{drt}')
                plt.xlim(0,40)
                plt.ylim(0,10000)
        plt.show()

    def delta_vs_NDVI_one_pixel(self):
        df = T.load_df(self.dff)
        outdir = join(self.this_class_png, 'delta_vs_NDVI_one_pixel')
        T.mk_dir(outdir,force=True)
        # T.open_path_and_file(outdir)
        # exit()
        # T.print_head_n(df)
        # exit()
        eli_list = ['Energy-Limited', 'Water-Limited']
        drt_list = ['hot-drought', 'normal-drought']
        drought_season_list = global_drought_season_list
        for season in drought_season_list:
            for eli in eli_list:
                for drt in drt_list:
                    df_season = df[df['drought_season'] == season]
                    df_eli = df_season[df_season['ELI_class'] == eli]
                    df_drt = df_eli[df_eli['drought_type'] == drt]

                    # delta_temp = df_pix['delta_temp'].tolist()
                    # rt_list = df_pix['rt'].tolist()
                    delta_temp_bins = np.linspace(-20, 20, 100)
                    df_group, bins_list_str = T.df_bin(df_drt,'delta_temp',delta_temp_bins)
                    y_list = []
                    x_list = []
                    err_list = []
                    for name,df_group_i in df_group:
                        left = name[0].left
                        vals = df_group_i['rt'].tolist()
                        err = np.nanstd(vals)
                        mean = np.nanmean(vals)
                        x_list.append(left)
                        y_list.append(mean)
                        err_list.append(err)
                    y_list = np.array(y_list)
                    err_list = np.array(err_list)
                    plt.plot(x_list, y_list)
                    plt.fill_between(x_list, y_list - err_list, y_list + err_list, alpha=0.5)
                    plt.title(f'{eli}-{drt}-{season}')
                    # plt.xticks(rotation=90)
                    plt.hlines(1, -20, 20, 'r')
                    plt.vlines(0, 0.9, 1.1, 'k')
                    plt.tight_layout()
                    outf = join(outdir, f'{eli}-{drt}-{season}.png')
                    plt.savefig(outf)
                    plt.close()
        T.open_path_and_file(outdir)


    def timing_trajectory_Tair(self):
        import analysis
        df = T.load_df(self.dff)
        outdir = join(self.this_class_png, 'timing_trajectory_Tair_mon_opt')
        T.mk_dir(outdir)
        timing_list = global_drought_season_list
        ELI_class_list = global_ELI_class_list
        drought_type_list = global_drought_type_list
        T.print_head_n(df)
        Topt_dir = join(analysis.Optimal_temperature_monthly().this_class_tif, r'optimal_temperature_mean')
        opt_temp_spatial_dict_all = {}
        for mon in range(1, 13):
            fpath = join(Topt_dir, f'optimal_temperature_{mon:02d}.tif')
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            opt_temp_spatial_dict_all[mon] = spatial_dict

        season_list = []

        for i,row in df.iterrows():
            mon = row['drought_mon']
            season = global_season_mon_dict[mon]
            season_list.append(season)
        df['drought_season'] = season_list

        for timing in timing_list:
            for ELI_class in ELI_class_list:
                plt.figure(figsize=(9*centimeter_factor,6*centimeter_factor))
                for drt in drought_type_list:
                    df_ELI = df[df['ELI_class'] == ELI_class]
                    df_drt = df_ELI[df_ELI['drought_type'] == drt]
                    df_timing = df_drt[df_drt['drought_season'] == timing]
                    T.print_head_n(df_timing)
                    pix_list = df_timing['pix'].tolist()
                    opt_temp_mean_list_mean_list = []
                    for mon in global_gs:
                        opt_temp_mean_list = []
                        for pix in pix_list:
                            opt_temp_mean = opt_temp_spatial_dict_all[mon][pix]
                            opt_temp_mean_list.append(opt_temp_mean)
                        opt_temp_mean_list_mean = np.nanmean(opt_temp_mean_list)
                        opt_temp_mean_list_mean_list.append(opt_temp_mean_list_mean)
                    # plt.plot(opt_temp_mean_list_mean_list,label=f'{drt}')

                    NDVI_process = df_timing['Temperature-origin_progress'].tolist()
                    NDVI_process = np.array(NDVI_process)
                    NDVI_process_mean = np.nanmean(NDVI_process,axis=0)
                    NDVI_process_mean = NDVI_process_mean[:3*6]
                    NDVI_process_mean = np.array(NDVI_process_mean)
                    opt_temp_mean_list_mean_list_3 = opt_temp_mean_list_mean_list*3
                    opt_temp_mean_list_mean_list_3 = np.array(opt_temp_mean_list_mean_list_3)
                    delta = NDVI_process_mean - opt_temp_mean_list_mean_list_3
                    plt.plot(delta,label=f'{drt}')
                    # plt.plot(NDVI_process_mean,label=f'{drt}')
                    # plt.plot(opt_temp_mean_list_mean_list*3,label=f'{drt}_optimal_temp')
                    # plt.show()
                plt.legend()
                plt.title(f'{timing}\n{ELI_class}')
                # plt.ylim(0,26)
                plt.ylim(-5,5)
                plt.grid()
                plt.xticks(list(range(len(NDVI_process_mean)))[::6],[-1,0,1])
                fname = f'{timing}_{ELI_class}.pdf'
                outf = join(outdir,fname)
                plt.savefig(outf)
                plt.close()
                # plt.show()
        T.open_path_and_file(outdir)

        pass



    def T_NDVI_relation(self):
        import analysis
        df = T.load_df(self.dff)
        outdir = join(self.this_class_png, 'T_NDVI_relation')
        T.mk_dir(outdir)

        AI_class_list = global_AI_class_list
        # Temp_dict, Temp_name = Load_Data().Temperature_origin()
        # Temp_dict, Temp_name = Load_Data().Temperature_anomaly_detrend()
        # Temp_dict_f = join(analysis.Optimal_temperature().this_class_arr,'T_vs_optimal_temp_delta/T_vs_optimal_temp_delta.npy')
        Temp_dict_f = join(analysis.Optimal_temperature_monthly().this_class_arr,r'T_vs_optimal_temp_delta\T_vs_optimal_temp_delta.npy')
        Temp_dict = T.load_npy(Temp_dict_f)
        # NDVI_dict, NDVI_name = Load_Data().NDVI_origin()
        NDVI_dict, NDVI_name = Load_Data().NDVI_anomaly_detrend()
        for AI in AI_class_list:
            df_AI = df[df['AI_class'] == AI]
            # pix_list = df_AI['pix'].tolist()
            # pix_list = list(set(pix_list))
            # Temp_list = []
            # NDVI_list = []
            # for pix in tqdm(pix_list):
            #     NDVI = NDVI_dict[pix]
            #     Temp = Temp_dict[pix]
            #     NDVI = np.array(NDVI)
            #     Temp = np.array(Temp)
            #     NDVI[NDVI>10000] = np.nan
            #     NDVI[NDVI<0] = np.nan
            #
            #     Temp[Temp > 10000] = np.nan
            #     Temp[Temp < -9999] = np.nan
            #     NDVI_gs_array = T.monthly_vals_to_annual_val(NDVI,method='array',grow_season=global_gs)
            #     NDVI_gs_array_flatten = NDVI_gs_array.flatten()
            #     Temp_gs_array = T.monthly_vals_to_annual_val(Temp,method='array',grow_season=global_gs)
            #     Temp_gs_array_flatten = Temp_gs_array.flatten()
            #     NDVI = NDVI_gs_array_flatten.tolist()
            #     Temp = Temp_gs_array_flatten.tolist()
            #
            #     Temp_list.extend(Temp)
            #     NDVI_list.extend(NDVI)
            # df_i = pd.DataFrame()
            # df_i['NDVI'] = NDVI_list
            # df_i['Temp'] = Temp_list
            # temp_bins = np.linspace(0,25,50)
            # df_group, bins_list_str = T.df_bin(df_i,'Temp',temp_bins)
            # x = []
            # y = []
            # for name,df_group_i in df_group:
            #     vals = df_group_i['NDVI'].tolist()
            #     mean = np.nanmean(vals)
            #     left = name[0].left
            #     x.append(left)
            #     y.append(mean)
            # plt.plot(x,y,label=f'{AI}')
            # plt.hist(Temp_list,bins=100,alpha=0.5,label=f'{AI}')
            # plt.show()
            plt.figure()
            drought_season_list = global_drought_season_list
            drought_season_index_dict = {
                'spring':(0,1),
                'summer':(2,3),
                'autumn':(4,5),
            }
            drt_color_dict = {
                'hot-drought':'r',
                'normal-drought':'b',
            }
            for season in drought_season_list:
                df_season = df_AI[df_AI['drought_season'] == season]
                drt_list = global_drought_type_list
                for drt in drt_list:

                    df_drt = df_season[df_season['drought_type'] == drt]
                    Temp_list_drought_year = []
                    NDVI_list_drought_year = []
                    for i,row in tqdm(df_drt.iterrows(),total=len(df_drt)):
                        pix = row['pix']
                        if not pix in Temp_dict:
                            continue
                        drought_year = row['drought_year']
                        NDVI = NDVI_dict[pix]
                        Temp = Temp_dict[pix]
                        NDVI = np.array(NDVI)
                        Temp = np.array(Temp)
                        NDVI[NDVI > 10000] = np.nan
                        NDVI[NDVI < -10000] = np.nan

                        Temp[Temp > 10000] = np.nan
                        Temp[Temp < -9999] = np.nan
                        # print(NDVI)
                        # print(np.shape(NDVI))
                        # plt.plot(NDVI)
                        # plt.show()
                        NDVI_gs_array = T.monthly_vals_to_annual_val(NDVI, method='array', grow_season=global_gs)
                        NDVI_gs_array_drought_year = NDVI_gs_array[drought_year-1982]

                        Temp_gs_array = T.monthly_vals_to_annual_val(Temp, method='array', grow_season=global_gs)
                        Temp_gs_array_drought_year = Temp_gs_array[drought_year-1982]

                        NDVI_gs_array_drought_year = NDVI_gs_array_drought_year.tolist()[drought_season_index_dict[season][0]:drought_season_index_dict[season][1]+1]
                        try:
                            Temp_gs_array_drought_year = Temp_gs_array_drought_year.tolist()[drought_season_index_dict[season][0]:drought_season_index_dict[season][1]+1]
                        except:
                            # print(drought_year)
                            # print(Temp_gs_array_drought_year)
                            # print(Temp_gs_array)
                            continue

                        # print(Temp_gs_array_drought_year)
                        # exit()

                        Temp_list_drought_year.extend(Temp_gs_array_drought_year)
                        NDVI_list_drought_year.extend(NDVI_gs_array_drought_year)
                    # exit()
                    df_j = pd.DataFrame()
                    df_j['NDVI'] = NDVI_list_drought_year
                    df_j['Temp'] = Temp_list_drought_year
                    # plt.hist(Temp_list_drought_year, bins=100, alpha=0.5, label=f'{season} {drt}')
                    # plt.show()
                    # df_j = df_j.dropna()
                    print(len(df_j))

                    # temp_bins = np.linspace(-10, 40, 46)
                    temp_bins = np.linspace(-10, 10, 21)
                    df_group, bins_list_str = T.df_bin(df_j, 'Temp', temp_bins)
                    NDVI_vals = df_j['NDVI'].tolist()
                    NDVI_vals = np.array(NDVI_vals)
                    Temp_vals = df_j['Temp'].tolist()
                    Temp_vals = np.array(Temp_vals)
                    # plt.scatter(Temp_vals,NDVI_vals,c=drt_color_dict[drt],alpha=0.3,s=1)
                    x = []
                    y = []
                    err = []
                    for name, df_group_i in df_group:
                        vals = df_group_i['NDVI'].tolist()
                        if len(vals) == 0:
                            mean = np.nan
                            err_i = np.nan
                        else:
                            mean = np.nanmean(vals)
                            # err_i = np.nanstd(vals)
                            err_i,_,_ = T.uncertainty_err(vals)
                        left = name[0].left
                        x.append(left)
                        y.append(mean)
                        err.append(err_i)
                    x = np.array(x)
                    y = np.array(y)
                    err = np.array(err)
                    # y_smooth = SMOOTH().smooth_convolve(y, 11)
                    # plt.plot(x, y_smooth, label=f'{season} {drt}')
                    plt.plot(x, y, label=f'{season} {drt}',color=drt_color_dict[drt])
                    plt.fill_between(x, y - err, y + err, alpha=0.3, color=drt_color_dict[drt])
                    plt.ylim(-2, 1)
                plt.title(f'{AI} {season}')
                plt.legend()
                outf = join(outdir, f'{AI}_{season}.pdf')
                # plt.savefig(outf)
                # plt.close()
                plt.show()
        T.open_path_and_file(outdir)

    def T_NDVI_relation_hist(self):
        import analysis
        df = T.load_df(self.dff)
        outdir = join(self.this_class_png, 'T_NDVI_relation_hist')
        T.mk_dir(outdir)

        AI_class_list = global_AI_class_list
        # Temp_dict, Temp_name = Load_Data().Temperature_origin()
        # Temp_dict, Temp_name = Load_Data().Temperature_anomaly_detrend()
        # Temp_dict_f = join(analysis.Optimal_temperature().this_class_arr,'T_vs_optimal_temp_delta/T_vs_optimal_temp_delta.npy')
        Temp_dict_f = join(analysis.Optimal_temperature_monthly().this_class_arr,r'T_vs_optimal_temp_delta\T_vs_optimal_temp_delta.npy')
        Temp_dict = T.load_npy(Temp_dict_f)
        # NDVI_dict, NDVI_name = Load_Data().NDVI_origin()
        NDVI_dict, NDVI_name = Load_Data().NDVI_anomaly_detrend()
        for AI in AI_class_list:
            df_AI = df[df['AI_class'] == AI]
            plt.figure()
            drought_season_list = global_drought_season_list
            drought_season_index_dict = {
                'spring':(0,1),
                'summer':(2,3),
                'autumn':(4,5),
            }
            drt_color_dict = {
                'hot-drought':'r',
                'normal-drought':'b',
            }
            for season in drought_season_list:
                df_season = df_AI[df_AI['drought_season'] == season]
                drt_list = global_drought_type_list
                for drt in drt_list:

                    df_drt = df_season[df_season['drought_type'] == drt]
                    Temp_list_drought_year = []
                    NDVI_list_drought_year = []
                    for i,row in tqdm(df_drt.iterrows(),total=len(df_drt)):
                        pix = row['pix']
                        if not pix in Temp_dict:
                            continue
                        drought_year = row['drought_year']
                        NDVI = NDVI_dict[pix]
                        Temp = Temp_dict[pix]
                        NDVI = np.array(NDVI)
                        Temp = np.array(Temp)
                        NDVI[NDVI > 10000] = np.nan
                        NDVI[NDVI < -10000] = np.nan

                        Temp[Temp > 10000] = np.nan
                        Temp[Temp < -9999] = np.nan
                        # print(NDVI)
                        # print(np.shape(NDVI))
                        # plt.plot(NDVI)
                        # plt.show()
                        NDVI_gs_array = T.monthly_vals_to_annual_val(NDVI, method='array', grow_season=global_gs)
                        NDVI_gs_array_drought_year = NDVI_gs_array[drought_year-1982]

                        Temp_gs_array = T.monthly_vals_to_annual_val(Temp, method='array', grow_season=global_gs)
                        Temp_gs_array_drought_year = Temp_gs_array[drought_year-1982]

                        NDVI_gs_array_drought_year = NDVI_gs_array_drought_year.tolist()[drought_season_index_dict[season][0]:drought_season_index_dict[season][1]+1]
                        Temp_gs_array_drought_year = Temp_gs_array_drought_year.tolist()[drought_season_index_dict[season][0]:drought_season_index_dict[season][1]+1]

                        # print(Temp_gs_array_drought_year)
                        # exit()

                        Temp_list_drought_year.extend(Temp_gs_array_drought_year)
                        NDVI_list_drought_year.extend(NDVI_gs_array_drought_year)
                    # exit()
                    df_j = pd.DataFrame()
                    df_j['NDVI'] = NDVI_list_drought_year
                    df_j['Temp'] = Temp_list_drought_year
                    # plt.hist(Temp_list_drought_year, bins=100, alpha=0.5, label=f'{season} {drt}')
                    # plt.show()
                    # df_j = df_j.dropna()
                    print(len(df_j))

                    # temp_bins = np.linspace(-10, 40, 46)
                    temp_bins = np.linspace(-10, 10, 21)
                    df_group, bins_list_str = T.df_bin(df_j, 'Temp', temp_bins)
                    NDVI_vals = df_j['NDVI'].tolist()
                    NDVI_vals = np.array(NDVI_vals)
                    Temp_vals = df_j['Temp'].tolist()
                    Temp_vals = np.array(Temp_vals)
                    # plt.hist(Temp_vals,bins=100, alpha=0.5, label=f'{season} {drt}')
                    x,y = Plot().plot_hist_smooth(Temp_vals,bins=100,alpha=0,range=(-10,10))
                    plt.plot(x, y, label=f'{season} {drt}',color=drt_color_dict[drt])
                    # plt.fill_between(x, y - err, y + err, alpha=0.3, color=drt_color_dict[drt])
                    # plt.ylim(-2, 1)
                plt.title(f'{AI} {season}')
                plt.legend()
                outf = join(outdir, f'{AI}_{season}.pdf')
                plt.savefig(outf)
                plt.close()
                # plt.show()
        T.open_path_and_file(outdir)

    def T_NDVI_relation_CSIF(self):
        import analysis
        df = T.load_df(self.dff)
        # outdir = join(self.this_class_png, 'T_NDVI_relation_CSIF')
        outdir = join(self.this_class_png, 'T_NDVI_relation_NDVI')
        T.mk_dir(outdir)

        AI_class_list = global_AI_class_list
        # Temp_dict, Temp_name = Load_Data().Temperature_origin()
        # Temp_dict, Temp_name = Load_Data().Temperature_anomaly_detrend()
        # Temp_dict_f = join(analysis.Optimal_temperature().this_class_arr,'T_vs_optimal_temp_delta/T_vs_optimal_temp_delta.npy')
        # Temp_dict_f = join(analysis.Optimal_temperature_monthly().this_class_arr,r'T_vs_optimal_temp_delta\T_vs_optimal_temp_delta.npy')
        Temp_dict_f = join(analysis.Optimal_temperature().this_class_arr,r'T_vs_optimal_temp_delta/TCSIF.npy')
        # Temp_dict_f = join(analysis.Optimal_temperature().this_class_arr,r'T_vs_optimal_temp_delta/T_vs_optimal_temp_delta.npy')
        Temp_dict = T.load_npy(Temp_dict_f)
        # NDVI_dict, NDVI_name = Load_Data().NDVI_origin()
        NDVI_dict, NDVI_name = Load_Data().NDVI_anomaly_detrend()
        for AI in AI_class_list:
            df_AI = df[df['AI_class'] == AI]
            plt.figure()
            drought_season_list = global_drought_season_list
            drought_season_index_dict = {
                'spring':(0,1),
                'summer':(2,3),
                'autumn':(4,5),
            }
            drt_color_dict = {
                'hot-drought':'r',
                'normal-drought':'b',
            }
            for season in drought_season_list:
                df_season = df_AI[df_AI['drought_season'] == season]
                drt_list = global_drought_type_list
                for drt in drt_list:

                    df_drt = df_season[df_season['drought_type'] == drt]
                    Temp_list_drought_year = []
                    NDVI_list_drought_year = []
                    for i,row in tqdm(df_drt.iterrows(),total=len(df_drt)):
                        pix = row['pix']
                        if not pix in Temp_dict:
                            continue
                        drought_year = row['drought_year']
                        NDVI = NDVI_dict[pix]
                        Temp = Temp_dict[pix]
                        NDVI = np.array(NDVI)
                        Temp = np.array(Temp)
                        # plt.plot(Temp)
                        # plt.show()
                        NDVI[NDVI > 10000] = np.nan
                        NDVI[NDVI < -10000] = np.nan

                        Temp[Temp > 10000] = np.nan
                        Temp[Temp < -9999] = np.nan
                        # print(NDVI)
                        # print(np.shape(NDVI))
                        # plt.plot(NDVI)
                        # plt.show()
                        NDVI_gs_array = T.monthly_vals_to_annual_val(NDVI, method='array', grow_season=global_gs)
                        NDVI_gs_array_drought_year = NDVI_gs_array[drought_year-1982]

                        Temp_gs_array = T.monthly_vals_to_annual_val(Temp, method='array', grow_season=global_gs)
                        Temp_gs_array_drought_year = Temp_gs_array[drought_year-1982]

                        NDVI_gs_array_drought_year_mean = np.nanmean(NDVI_gs_array_drought_year)
                        Temp_gs_array_drought_year_mean = np.nanmean(Temp_gs_array_drought_year)
                        # NDVI_gs_array_drought_year = NDVI_gs_array_drought_year.tolist()[drought_season_index_dict[season][0]:drought_season_index_dict[season][1]+1]
                        # print(NDVI_gs_array_drought_year)
                        # exit()
                        # try:
                        #     Temp_gs_array_drought_year = Temp_gs_array_drought_year.tolist()[drought_season_index_dict[season][0]:drought_season_index_dict[season][1]+1]
                        # except:
                        #     continue

                        Temp_list_drought_year.extend(Temp_gs_array_drought_year)
                        NDVI_list_drought_year.extend(NDVI_gs_array_drought_year)
                    # exit()
                    df_j = pd.DataFrame()
                    df_j['NDVI'] = NDVI_list_drought_year
                    df_j['Temp'] = Temp_list_drought_year
                    # plt.hist(Temp_list_drought_year, bins=100, alpha=0.5, label=f'{season} {drt}')
                    # plt.show()
                    # df_j = df_j.dropna()
                    print(len(df_j))

                    # temp_bins = np.linspace(-10, 40, 46)
                    temp_bins = np.linspace(-10, 10, 21)
                    df_group, bins_list_str = T.df_bin(df_j, 'Temp', temp_bins)
                    NDVI_vals = df_j['NDVI'].tolist()
                    NDVI_vals = np.array(NDVI_vals)
                    Temp_vals = df_j['Temp'].tolist()
                    Temp_vals = np.array(Temp_vals)
                    # plt.scatter(Temp_vals,NDVI_vals,c=drt_color_dict[drt],alpha=0.3,s=1)
                    x = []
                    y = []
                    err = []
                    for name, df_group_i in df_group:
                        vals = df_group_i['NDVI'].tolist()
                        if len(vals) == 0:
                            mean = np.nan
                            err_i = np.nan
                        else:
                            mean = np.nanmean(vals)
                            # err_i = np.nanstd(vals)
                            err_i,_,_ = T.uncertainty_err(vals)
                        left = name[0].left
                        x.append(left)
                        y.append(mean)
                        err.append(err_i)
                    x = np.array(x)
                    y = np.array(y)
                    err = np.array(err)
                    # y_smooth = SMOOTH().smooth_convolve(y, 11)
                    # plt.plot(x, y_smooth, label=f'{season} {drt}')
                    plt.plot(x, y, label=f'{season} {drt}',color=drt_color_dict[drt])
                    plt.fill_between(x, y - err, y + err, alpha=0.3, color=drt_color_dict[drt])
                    plt.ylim(-2, 1)
                plt.title(f'{AI} {season}')
                plt.legend()
                outf = join(outdir, f'{AI}_{season}.pdf')
                plt.savefig(outf)
                plt.close()
                # plt.show()
        # T.open_path_and_file(outdir)

def Load_dataframe():
    dff = Dataframe().dff
    df = T.load_df(dff)
    return df

def main():
    # Dataframe().run()
    Compensation_Excerbation().run()
    # Compensation_Excerbation_heatwave().run()
    # Drought_timing().run()
    # Random_Forests().run()
    # Random_Forests_delta().run()
    # Partial_Dependence_Plots().run()
    # Partial_Dependence_Plots_Delta().run()
    # Single_Variable_Analysis().run()
    # Phenology_Statistic().run()
    # Optimal_temperature_statistic().run()
    # SEM().run()


    pass


if __name__ == '__main__':
    main()
