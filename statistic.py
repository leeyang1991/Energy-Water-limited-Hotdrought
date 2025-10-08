# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np

from meta_info import *
result_root_this_script = join(results_root, 'statistic')

class Dataframe_func:

    def __init__(self,df,is_clean_df=True):
        print('add lon lat')
        df = self.add_lon_lat(df)
        df = self.add_gez(df)
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

    def add_AI_new_to_df(self, df):
        f = join(data_root, 'Aridity_index_new/ai_05_rescaled.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'aridity_index_new')
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

    def add_gez(self,df):
        f = join(data_root, 'Global_Ecological_Zone/tif/gez_2010.tif')
        legend_f = join(data_root, 'Global_Ecological_Zone/legend.npy')
        legend_dict = T.load_npy(legend_f)
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(f)
        spatial_dict_legend = {}
        for pix in tqdm(spatial_dict):
            val = spatial_dict[pix]
            if np.isnan(val):
                continue
            legend = legend_dict[val]
            spatial_dict_legend[pix] = legend
        df = T.add_spatial_dic_to_df(df, spatial_dict_legend, 'GEZ')
        return df

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
        df = self.__gen_df_init()
        # df = Dataframe_func(df).df
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
        self.delta_area_ratio_gradient_bar()

        # self.rs_rt_area_ratio_bar(df)
        # self.delta_area_ratio_bar()
        # self.delta_area_statistic()
        # self.delta_value_statistic()
        # self.print_delta_value_statistic()
        # self.check_compensation_excerbation_season()
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
        NDVI_spatial_dict,_,_ = Load_Data().NDVI_origin()
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
        # plt.savefig(outf)
        # plt.close()
        # T.open_path_and_file(outdir)
        plt.show()

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


class Drought_timing:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Drought_timing', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'Drought_timing.df')
        pass

    def run(self):
        # df = Over_shoot_phenology().add_NDVI_process()
        df = self.__gen_df_init()
        # df = Dataframe_func(df).df
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
        # df = self.add_longterm_growing_season(df)
        # T.print_head_n(df)
        # df = self.add_VPD_anomaly_process(df)
        # df = self.add_VPD_origin_process(df)
        # df = self.add_NDVI_percentage_process(df)
        # df = self.add_CSIF_percentage_process(df)
        # df = self.add_CSIF_anomaly_process(df)
        # df = self.add_GPP_NIRv_percentage_process(df)
        # df = self.add_GPP_NIRv_anomaly_process(df)
        # df = self.add_NIRv_percentage_process(df)
        # df = self.add_NIRv_anomaly_process(df)

        # T.save_df(df, self.dff)
        # T.df_to_excel(df, self.dff)

        # plot tifs
        # self.Drought_year_spatial_tif(df)
        # self.Drought_year_percentage_spatial_tif(df)
        # self.Drought_year_percentage_sig_spatial_tif()
        # self.plot_Drought_year_spatial_tif()

        # statistic
        # self.timing_trajectory(df)
        # self.timing_trajectory_1_year(df)
        # self.timing_trajectory_1_year_positive(df)
        # self.timing_trajectory_CSIF(df)
        # self.timing_trajectory_GPP_NIRv(df)
        # self.timing_trajectory_sm(df)
        # self.timing_trajectory_Tair(df)
        # self.delta_season_tif(df)
        # self.seasonal_ndvi_tif(df)
        # self.seasonal_ndvi_tif_statistic()
        # self.seasonal_ndvi_tif_area_statistic()
        # self.delta_season_bar(df)
        # self.delta_season_boxplot(df)

        self.season_excerbation_alleviation_ratio_tif(df)
        self.season_excerbation_alleviation_ratio_statistic()
        self.plot_season_excerbation_alleviation_ratio()

        # self.season_excerbation_alleviation_ratio_tif_CSIF(df)
        # self.season_excerbation_alleviation_ratio_statistic_CSIF()
        # self.plot_season_excerbation_alleviation_ratio_CSIF()

        # self.season_excerbation_alleviation_ratio_tif_GPP_NIRv(df)
        # self.season_excerbation_alleviation_ratio_statistic_GPP_NIRv()
        # self.plot_season_excerbation_alleviation_ratio_GPP_NIRv()

        # self.delta_season_bar_all(df)
        # self.delta_season_bar_all1()
        # self.delta_season_box_all(df)
        # self.delta_season_bar_ANOVA(df)
        # self.delta_season_bar_error_bar(df)
        # self.check_compensation_excerbation_season()
        # self.delta_tif(df)
        # self.GEZ_statistic()
        # self.high_latitude_hot_drought()
        # self.VPD_delta_tif(df)
        # self.VPD_alleviation_excerbation()
        # self.VPD_NDVI(df)

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

    def add_longterm_growing_season(self,df):
        import analysis
        # gs_f = join(analysis.Phenology().this_class_arr, 'longterm_growing_season', 'longterm_growing_season.npy')
        gs_f = join(analysis.Longterm_Phenology().this_class_arr, 'SOS_EOS', 'SOS_EOS.npy')
        gs_dict = T.load_npy(gs_f)
        df = T.add_spatial_dic_to_df(df,gs_dict,'growing_season')
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
        color_dict = {
            'normal-drought': 'b',
            'hot-drought': 'r',
        }

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
                    x_list = list(range(len(NDVI_process_mean)))
                    x_list = np.array(x_list)
                    x_list = np.insert(x_list,6,6.5)
                    x_list = np.insert(x_list,13,12.5)

                    NDVI_process_mean = np.insert(NDVI_process_mean,6,np.nan)
                    NDVI_process_std = np.insert(NDVI_process_std,6,np.nan)
                    NDVI_process_mean = np.insert(NDVI_process_mean,13,np.nan)
                    NDVI_process_std = np.insert(NDVI_process_std,13,np.nan)
                    plt.plot(x_list,NDVI_process_mean)
                    plt.scatter(x_list,NDVI_process_mean,marker='o',s=10,c=color_dict[drt],zorder=10,alpha=0.5)
                    plt.fill_between(x_list,NDVI_process_mean-NDVI_process_std,NDVI_process_mean+NDVI_process_std,alpha=0.3)
                # plt.legend()
                plt.title(f'{timing}\n{ELI_class}')
                plt.ylim(-1.1,0.5)
                # plt.grid()
                # plt.xticks(list(range(len(NDVI_process_mean)))[::6],[-1,0,1])
                fname = f'{timing}_{ELI_class}.pdf'
                outf = join(outdir,fname)
                plt.savefig(outf)
                plt.close()
                # plt.show()
        # T.open_path_and_file(outdir)

        pass
    def timing_trajectory_1_year(self,df):
        outdir = join(self.this_class_png, 'timing_trajectory_1_year')
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
        color_dict = {
            'normal-drought': 'b',
            'hot-drought': 'r',
        }

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
                    # plt.plot(NDVI_process_mean)
                    # plt.show()
                    NDVI_process_std = np.nanstd(NDVI_process,axis=0) / 6.
                    # NDVI_process_std = T.uncertainty_err_2d(NDVI_process,axis=0)
                    NDVI_process_mean = NDVI_process_mean[6:2*6]
                    NDVI_process_std = NDVI_process_std[6:2*6]
                    x_list = list(range(len(NDVI_process_mean)))
                    x_list = np.array(x_list)
                    # x_list = np.insert(x_list,6,6.5)
                    # x_list = np.insert(x_list,13,12.5)

                    # NDVI_process_mean = np.insert(NDVI_process_mean,6,np.nan)
                    # NDVI_process_std = np.insert(NDVI_process_std,6,np.nan)
                    # NDVI_process_mean = np.insert(NDVI_process_mean,13,np.nan)
                    # NDVI_process_std = np.insert(NDVI_process_std,13,np.nan)
                    plt.plot(x_list,NDVI_process_mean)
                    plt.scatter(x_list,NDVI_process_mean,marker='o',s=10,c=color_dict[drt],zorder=10,alpha=0.5)
                    plt.fill_between(x_list,NDVI_process_mean-NDVI_process_std,NDVI_process_mean+NDVI_process_std,alpha=0.3)
                # plt.legend()
                plt.title(f'{timing}\n{ELI_class}')
                plt.ylim(-1.1,0.5)
                # plt.grid()
                # plt.xticks(list(range(len(NDVI_process_mean)))[::6],[-1,0,1])
                fname = f'{timing}_{ELI_class}.pdf'
                outf = join(outdir,fname)
                plt.savefig(outf)
                plt.close()
                # plt.show()
        T.open_path_and_file(outdir)

        pass

    def timing_trajectory_1_year_positive(self,df):
        outdir = join(self.this_class_png, 'timing_trajectory_1_year_positive')
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
        color_dict = {
            'normal-drought': 'b',
            'hot-drought': 'r',
        }
        # df = df[df['rt'] > 1]
        # mode = 'positive'
        mode = 'all'
        drt = 'hot-drought'
        df_drt = df[df['drought_type'] == drt]

        for timing in timing_list:
            plt.figure(figsize=(9*centimeter_factor,6*centimeter_factor))
            # for drt in drought_type_list:
            df_timing = df_drt[df_drt['drought_season'] == timing]
            NDVI_process = df_timing['NDVI_progress'].tolist()
            NDVI_process = np.array(NDVI_process)
            NDVI_process_mean = np.nanmean(NDVI_process,axis=0)
            # plt.plot(NDVI_process_mean)
            # plt.show()
            NDVI_process_std = np.nanstd(NDVI_process,axis=0) / 6.
            # NDVI_process_std = T.uncertainty_err_2d(NDVI_process,axis=0)
            NDVI_process_mean = NDVI_process_mean[6:2*6]
            NDVI_process_std = NDVI_process_std[6:2*6]
            x_list = list(range(len(NDVI_process_mean)))
            x_list = np.array(x_list)
            # x_list = np.insert(x_list,6,6.5)
            # x_list = np.insert(x_list,13,12.5)

            # NDVI_process_mean = np.insert(NDVI_process_mean,6,np.nan)
            # NDVI_process_std = np.insert(NDVI_process_std,6,np.nan)
            # NDVI_process_mean = np.insert(NDVI_process_mean,13,np.nan)
            # NDVI_process_std = np.insert(NDVI_process_std,13,np.nan)
            plt.plot(x_list,NDVI_process_mean)
            plt.scatter(x_list,NDVI_process_mean,marker='o',s=10,c=color_dict[drt],zorder=10,alpha=0.5)
            plt.fill_between(x_list,NDVI_process_mean-NDVI_process_std,NDVI_process_mean+NDVI_process_std,alpha=0.3)
            # plt.legend()
            plt.title(f'{timing}-{mode}')
            # plt.ylim(-1.1,0.5)
            # plt.grid()
            # plt.xticks(list(range(len(NDVI_process_mean)))[::6],[-1,0,1])
            plt.figure(figsize=(9*centimeter_factor,6*centimeter_factor))
            plt.title(f'{timing}-{mode}')
            sos = df_timing['SOS'].tolist()
            plt.hist(sos,bins=100,range=(-20,20))
            # print(sos);exit()

            # plt.savefig(outf)
            # plt.close()
        plt.show()
        # T.open_path_and_file(outdir)

        pass

    def timing_trajectory_CSIF(self,df):
        outdir = join(self.this_class_png, 'timing_trajectory_CSIF')
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
        color_dict = {
            'normal-drought': 'b',
            'hot-drought': 'r',
        }

        for timing in timing_list:
            for ELI_class in ELI_class_list:
                plt.figure(figsize=(9*centimeter_factor,6*centimeter_factor))
                for drt in drought_type_list:
                    df_ELI = df[df['ELI_class'] == ELI_class]
                    df_drt = df_ELI[df_ELI['drought_type'] == drt]
                    df_timing = df_drt[df_drt['drought_season'] == timing]
                    NDVI_process = df_timing['CSIF-anomaly_detrend_progress'].tolist()
                    NDVI_process = np.array(NDVI_process)
                    NDVI_process_mean = np.nanmean(NDVI_process,axis=0)
                    NDVI_process_std = np.nanstd(NDVI_process,axis=0) / 6.
                    # NDVI_process_std = T.uncertainty_err_2d(NDVI_process,axis=0)
                    NDVI_process_mean = NDVI_process_mean[:3*6]
                    NDVI_process_std = NDVI_process_std[:3*6]
                    x_list = list(range(len(NDVI_process_mean)))
                    x_list = np.array(x_list)
                    x_list = np.insert(x_list,6,6.5)
                    x_list = np.insert(x_list,13,12.5)

                    NDVI_process_mean = np.insert(NDVI_process_mean,6,np.nan)
                    NDVI_process_std = np.insert(NDVI_process_std,6,np.nan)
                    NDVI_process_mean = np.insert(NDVI_process_mean,13,np.nan)
                    NDVI_process_std = np.insert(NDVI_process_std,13,np.nan)
                    plt.plot(x_list,NDVI_process_mean)
                    plt.scatter(x_list,NDVI_process_mean,marker='o',s=10,c=color_dict[drt],zorder=10,alpha=0.5)
                    plt.fill_between(x_list,NDVI_process_mean-NDVI_process_std,NDVI_process_mean+NDVI_process_std,alpha=0.3)
                # plt.legend()
                plt.title(f'{timing}\n{ELI_class}')
                plt.ylim(-1.1,0.5)
                # plt.grid()
                # plt.xticks(list(range(len(NDVI_process_mean)))[::6],[-1,0,1])
                fname = f'{timing}_{ELI_class}.pdf'
                outf = join(outdir,fname)
                plt.savefig(outf)
                plt.close()
                # plt.show()
        # T.open_path_and_file(outdir)

        pass

    def timing_trajectory_GPP_NIRv(self,df):
        outdir = join(self.this_class_png, 'timing_trajectory_GPP_NIRv')
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
        color_dict = {
            'normal-drought': 'b',
            'hot-drought': 'r',
        }

        for timing in timing_list:
            for ELI_class in ELI_class_list:
                plt.figure(figsize=(9*centimeter_factor,6*centimeter_factor))
                for drt in drought_type_list:
                    df_ELI = df[df['ELI_class'] == ELI_class]
                    df_drt = df_ELI[df_ELI['drought_type'] == drt]
                    df_timing = df_drt[df_drt['drought_season'] == timing]
                    NDVI_process = df_timing['GPP_NIRv-anomaly_progress'].tolist()
                    NDVI_process = np.array(NDVI_process)
                    NDVI_process_mean = np.nanmean(NDVI_process,axis=0)
                    NDVI_process_std = np.nanstd(NDVI_process,axis=0) / 6.
                    # NDVI_process_std = T.uncertainty_err_2d(NDVI_process,axis=0)
                    NDVI_process_mean = NDVI_process_mean[:3*6]
                    NDVI_process_std = NDVI_process_std[:3*6]
                    x_list = list(range(len(NDVI_process_mean)))
                    x_list = np.array(x_list)
                    x_list = np.insert(x_list,6,6.5)
                    x_list = np.insert(x_list,13,12.5)

                    NDVI_process_mean = np.insert(NDVI_process_mean,6,np.nan)
                    NDVI_process_std = np.insert(NDVI_process_std,6,np.nan)
                    NDVI_process_mean = np.insert(NDVI_process_mean,13,np.nan)
                    NDVI_process_std = np.insert(NDVI_process_std,13,np.nan)
                    plt.plot(x_list,NDVI_process_mean)
                    plt.scatter(x_list,NDVI_process_mean,marker='o',s=10,c=color_dict[drt],zorder=10,alpha=0.5)
                    plt.fill_between(x_list,NDVI_process_mean-NDVI_process_std,NDVI_process_mean+NDVI_process_std,alpha=0.3)
                # plt.legend()
                plt.title(f'{timing}\n{ELI_class}')
                plt.ylim(-1.1,0.5)
                # plt.grid()
                # plt.xticks(list(range(len(NDVI_process_mean)))[::6],[-1,0,1])
                fname = f'{timing}_{ELI_class}.pdf'
                outf = join(outdir,fname)
                plt.savefig(outf)
                plt.close()
                # plt.show()
        # T.open_path_and_file(outdir)

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

    def delta_tif(self,df):
        outdir = join(self.this_class_tif, 'delta')
        T.mk_dir(outdir)
        drought_season_list = global_drought_season_list
        # print(df_season)
        pix_list = T.get_df_unique_val_list(df, 'pix')
        # print(len(pix_list))
        drought_type_list = global_drought_type_list

        # drought_year_list = range(1, 5)
        drought_year_list = [1]
        for drought_year_i in drought_year_list:
            df_group_dict = T.df_groupby(df, 'pix')
            spatial_dict = {}
            for pix in tqdm(pix_list):
                df_pix = df_group_dict[pix]
                df_hot = df_pix[df_pix['drought_type'] == 'hot-drought']
                df_normal = df_pix[df_pix['drought_type'] == 'normal-drought']
                if len(df_hot) == 0 or len(df_normal) == 0:
                    continue
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
            outf = join(outdir, f'delta.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict, outf)

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


    def seasonal_ndvi_tif(self,df):
        outdir = join(self.this_class_tif, 'seasonal_ndvi')
        T.mk_dir(outdir)
        T.print_head_n(df)
        drought_season_list = global_drought_season_list
        drought_type_list = global_drought_type_list
        ELI_class_list = global_ELI_class_list

        for season in drought_season_list:
            df_season = df[df['drought_season'] == season]
            for drt in drought_type_list:
                df_drt = df_season[df_season['drought_type'] == drt]
                for ELI in ELI_class_list:
                    df_ELI = df_drt[df_drt['ELI_class'] == ELI]
                    pix_list = T.get_df_unique_val_list(df_ELI, 'pix')
                    df_group_dict = T.df_groupby(df_ELI, 'pix')
                    spatial_dict = {}
                    for pix in tqdm(pix_list):
                        df_pix = df_group_dict[pix]
                        rt = df_pix['rt'].tolist()
                        rt_mean = np.nanmean(rt)
                        spatial_dict[pix] = rt_mean
                    outf = join(outdir, f'{season}_{drt}_{ELI}.tif')
                    DIC_and_TIF().pix_dic_to_tif(spatial_dict, outf)
        T.open_path_and_file(outdir)

    def seasonal_ndvi_tif_statistic(self):
        fdir = join(self.this_class_tif, 'seasonal_ndvi')
        seasonal_ndvi_vals_dic = {}
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            key = f.replace('.tif', '')
            fpath = join(fdir, f)
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            vals = np.array(list(spatial_dict.values()))
            vals = vals[~np.isnan(vals)]
            seasonal_ndvi_vals_dic[key] = vals
        for key in seasonal_ndvi_vals_dic:
            print(key)
            vals = seasonal_ndvi_vals_dic[key]
            print(np.mean(vals))

    def seasonal_ndvi_tif_area_statistic(self):
        fdir = join(self.this_class_tif, 'seasonal_ndvi')
        seasonal_ndvi_vals_dic = {}
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            # if not 'spring_hot-drought_Energy-Limited' in f:
            # if not 'summer_hot-drought_Energy-Limited' in f:
            if not 'autumn_hot-drought_Energy-Limited' in f:
                continue
            # print(f)
            key = f.replace('.tif', '')
            fpath = join(fdir, f)
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            vals = []
            for pix in spatial_dict:
                val = spatial_dict[pix]
                if np.isnan(val):
                    continue
                vals.append(val)
            vals = np.array(vals)
            vals_postive = vals>1.0
            pos_count = 0
            for t in vals_postive:
                if t:
                    pos_count += 1
            pos_ratio = pos_count / len(vals)
            print('pos_ratio',pos_ratio)
            print('mean',np.nanmean(vals))


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

    def delta_season_boxplot(self,df):
        # T.print_head_n(df)
        outdir = join(self.this_class_png, 'delta_season_boxplot')
        T.mk_dir(outdir)
        drought_season_list = global_drought_season_list
        ELI_class_list = global_ELI_class_list[::-1]

        boxes = []
        labels = []
        box_dict = {}
        for ELI_class in ELI_class_list:
            df_ELI = df[df['ELI_class'] == ELI_class]
            for season in drought_season_list:
                df_season = df_ELI[df_ELI['drought_season'] == season]
                # T.print_head_n(df_season)
                df_pix_dict = T.df_groupby(df_season, 'pix')
                delta_spatial_dict = {}
                vals_list = []
                for pix in tqdm(df_pix_dict):
                    df_pix = df_pix_dict[pix]
                    df_hot = df_pix[df_pix['drought_type'] == 'hot-drought']
                    df_normal = df_pix[df_pix['drought_type'] == 'normal-drought']
                    rt_hot = df_hot['rt'].tolist()
                    rt_normal = df_normal['rt'].tolist()
                    rt_hot_mean = np.nanmean(rt_hot)
                    rt_normal_mean = np.nanmean(rt_normal)
                    delta = rt_hot_mean - rt_normal_mean
                    delta_spatial_dict[pix] = delta
                    if not np.isnan(delta):
                        vals_list.append(delta)
                vals = np.array(vals_list)
                label = f'{ELI_class}_{season}'
                boxes.append(vals)
                labels.append(label)
                box_dict[label] = vals
        print(labels) # ['Water-Limited_spring', 'Water-Limited_summer', 'Water-Limited_autumn', 'Energy-Limited_spring', 'Energy-Limited_summer', 'Energy-Limited_autumn']

        season_color_dict = {
            'spring': 'g',
            'summer': 'r',
            'autumn': 'b',
        }
        pos = 0
        for season in drought_season_list:
            for ELI_class in ELI_class_list:
                label = f'{ELI_class}_{season}'
                vals = box_dict[label]
                x,y = Plot().plot_hist_smooth(vals,bins=30,alpha=0,range=(-0.3,0.3))
                if ELI_class == 'Water-Limited':
                    lw = 1
                else:
                    lw = 2
                # plt.boxplot(vals, labels=[label],positions=[pos], showfliers=False)
                # plt.plot(x,y,label=label,lw=lw,color=season_color_dict[season])
                plt.boxplot(vals, labels=[label],positions=[pos], showfliers=False,vert=False)
                pos += 1

        # plt.legend()
        plt.tight_layout()
        outf = join(outdir, f'delta_season_boxplot.pdf')
        plt.savefig(outf)
        # plt.show()
        exit()
        plt.boxplot(boxes, labels=labels, showfliers=False)
        plt.xticks(rotation=90)
        plt.ylim(-0.5, 0.5)
        plt.tight_layout()
        plt.show()
            #     arr = DIC_and_TIF().pix_dic_to_spatial_arr(delta_spatial_dict)
            #     plt.figure()
            #     plt.imshow(arr, cmap='RdBu', vmin=-0.2, vmax=0.2, interpolation='nearest')
            #     plt.title(season)
            #     plt.colorbar()
            # plt.show()

        pass
    def season_excerbation_alleviation_ratio_tif(self,df):
        # T.print_head_n(df)
        outdir = join(self.this_class_tif, 'season_excerbation_alleviation_ratio')
        T.mk_dir(outdir)
        global_drought_season_index_dict = {
            'spring': [0,1],
            'summer': [2,3],
            'autumn': [4,5],
        }
        drought_season_list = global_drought_season_list
        ELI_class_list = global_ELI_class_list[::-1]
        alleviation_list = []
        excerbation_list = []
        progress_col_name = 'NDVI-percentage_progress'

        for season in drought_season_list:
            df_season = df[df['drought_season'] == season]
            df_pix_dict = T.df_groupby(df_season, 'pix')
            spatial_dict_normal = {}
            spatial_dict_hot = {}
            for pix in tqdm(df_pix_dict):
                df_pix = df_pix_dict[pix]
                df_hot = df_pix[df_pix['drought_type'] == 'hot-drought']
                df_normal = df_pix[df_pix['drought_type'] == 'normal-drought']
                NDVI_progress_hot = df_hot[progress_col_name].tolist()
                NDVI_progress_normal = df_normal[progress_col_name].tolist()
                if len(NDVI_progress_hot) == 0 or len(NDVI_progress_normal) == 0:
                    continue
                NDVI_progress_hot_mean = np.nanmean(NDVI_progress_hot,axis=0)
                NDVI_progress_hot_mean_reshape = np.reshape(NDVI_progress_hot_mean,(6,-1))
                drought_season_vals_hot = NDVI_progress_hot_mean_reshape[1][global_drought_season_index_dict[season]]
                drought_season_vals_hot_mean = np.nanmean(drought_season_vals_hot)

                NDVI_progress_normal_mean = np.nanmean(NDVI_progress_normal,axis=0)
                NDVI_progress_normal_mean_reshape = np.reshape(NDVI_progress_normal_mean,(6,-1))
                drought_season_vals_normal = NDVI_progress_normal_mean_reshape[1][global_drought_season_index_dict[season]]
                drought_season_vals_normal_mean = np.nanmean(drought_season_vals_normal)

                # delta = drought_season_vals_hot_mean - drought_season_vals_normal_mean

                # spatial_dict[pix] = delta
                spatial_dict_hot[pix] = drought_season_vals_hot_mean
                spatial_dict_normal[pix] = drought_season_vals_normal_mean
            arr_hot = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_hot)
            outf_hot = join(outdir, f'hot_{season}.tif')
            DIC_and_TIF().arr_to_tif(arr_hot, outf_hot)

            arr_normal = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_normal)
            outf_normal = join(outdir, f'normal_{season}.tif')
            DIC_and_TIF().arr_to_tif(arr_normal, outf_normal)

        pass

    def season_excerbation_alleviation_ratio_statistic(self):
        fdir = join(self.this_class_tif, 'season_excerbation_alleviation_ratio')
        outdir = join(self.this_class_arr, 'season_excerbation_alleviation_ratio_statistic')
        drt_list = ['hot', 'normal']
        T.mk_dir(outdir)
        spatial_dict_all = {}
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir, f)
            # arr = DIC_and_TIF().spatial_tif_to_arr(fpath)
            # plt.imshow(arr, cmap='RdBu', vmin=-0.2, vmax=0.2, interpolation='nearest')
            # plt.colorbar()
            # plt.show()
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dict_mode = {}
            for pix in spatial_dict:
                val = spatial_dict[pix]
                if np.isnan(val):
                    continue

                if val > 5:
                    mode = 'alleviation'
                elif val < -5:
                    mode = 'excerbation'
                else:
                    mode = 'normal'
                    # raise ValueError
                spatial_dict_mode[pix] = mode
            key = f.replace('.tif', '')
            spatial_dict_all[key] = spatial_dict_mode
        df = T.spatial_dics_to_df(spatial_dict_all)
        df = Dataframe_func(df).df
        # T.print_head_n(df);exit()
        T.print_head_n(df)
        ratio_dict = {}
        ELI_class_list = global_ELI_class_list
        flag = 0
        for drt in drt_list:
            for season in global_drought_season_list:
                col_name = f'{drt}_{season}'
                df_season = df.dropna(subset=[col_name])
                for ELI_class in ELI_class_list:
                    df_ELI = df_season[df_season['ELI_class'] == ELI_class]
                    mode_list = T.get_df_unique_val_list(df_ELI, col_name)
                    # print(mode_list)
                    for mode in mode_list:
                        df_mode = df_ELI[df_ELI[col_name] == mode]
                        count = len(df_mode)
                        ratio = count / len(df_ELI)
                        # print(season, ELI_class, mode, count, f'{ratio:.3f}%')
                        ratio_dict[flag] = {
                            'season': season,
                            'drt': drt,
                            'ELI_class': ELI_class,
                            'mode': mode,
                            'count': count,
                            'ratio': f'{ratio*100:.3f}%'
                        }
                        flag += 1
        df_result = T.dic_to_df(ratio_dict, 'flag')
        T.print_head_n(df_result)
        outf = join(outdir, 'ratio')
        T.df_to_excel(df_result, outf)

        pass

    def plot_season_excerbation_alleviation_ratio(self):
        fpath = join(self.this_class_arr, 'season_excerbation_alleviation_ratio_statistic', 'ratio.xlsx')
        outdir = join(self.this_class_png, 'plot_season_excerbation_alleviation_ratio')
        T.mk_dir(outdir)
        df = pd.read_excel(fpath, index_col=0)
        T.print_head_n(df)
        exit()
        drt_list = ['hot', 'normal']
        ELI_class_list = global_ELI_class_list
        drought_season_list = global_drought_season_list
        mode_list = ['alleviation', 'excerbation']

        for ELI in ELI_class_list:
            plt.figure(figsize=(5, 5))
            df_ELI = df[df['ELI_class'] == ELI]
            for season in drought_season_list:
                df_season = df_ELI[df_ELI['season'] == season]
                for drt in drt_list[::-1]:
                    df_drt = df_season[df_season['drt'] == drt]
                    for mode in mode_list:
                        df_mode = df_drt[df_drt['mode'] == mode]
                        print(df_mode)
                        if len(df_mode) != 1:
                            raise
                        ratio_str = df_mode['ratio'].tolist()[0]
                        ratio = ratio_str.split('%')[0]
                        ratio = float(ratio)
                        color = 'blue'
                        if mode == 'excerbation':
                            ratio = -ratio
                            color = 'red'
                        plt.bar(f'{season} {drt}', ratio, color=color)
                        # key = f'{season} {ELI} {drt} {mode}'
                        # bar_dict[key] = {'ratio':ratio,'color':color}
            # pprint(bar_dict)
            plt.title(f'{ELI}')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.ylim(-95, 40)
            outf = join(outdir, f'{ELI}.pdf')
            # plt.savefig(outf, dpi=300)
            # plt.close()
            plt.show()

    def season_excerbation_alleviation_ratio_tif_CSIF(self,df):
        # T.print_head_n(df)
        outdir = join(self.this_class_tif, 'season_excerbation_alleviation_ratio_CSIF')
        T.mk_dir(outdir)
        global_drought_season_index_dict = {
            'spring': [0,1],
            'summer': [2,3],
            'autumn': [4,5],
        }
        drought_season_list = global_drought_season_list
        ELI_class_list = global_ELI_class_list[::-1]
        alleviation_list = []
        excerbation_list = []
        progress_col_name = 'CSIF-percentage_progress'

        for season in drought_season_list:
            df_season = df[df['drought_season'] == season]
            df_pix_dict = T.df_groupby(df_season, 'pix')
            spatial_dict_normal = {}
            spatial_dict_hot = {}
            for pix in tqdm(df_pix_dict):
                df_pix = df_pix_dict[pix]
                df_hot = df_pix[df_pix['drought_type'] == 'hot-drought']
                df_normal = df_pix[df_pix['drought_type'] == 'normal-drought']
                NDVI_progress_hot = df_hot[progress_col_name].tolist()
                NDVI_progress_normal = df_normal[progress_col_name].tolist()
                if len(NDVI_progress_hot) == 0 or len(NDVI_progress_normal) == 0:
                    continue
                NDVI_progress_hot_mean = np.nanmean(NDVI_progress_hot,axis=0)
                NDVI_progress_hot_mean_reshape = np.reshape(NDVI_progress_hot_mean,(6,-1))
                drought_season_vals_hot = NDVI_progress_hot_mean_reshape[1][global_drought_season_index_dict[season]]
                drought_season_vals_hot_mean = np.nanmean(drought_season_vals_hot)

                NDVI_progress_normal_mean = np.nanmean(NDVI_progress_normal,axis=0)
                NDVI_progress_normal_mean_reshape = np.reshape(NDVI_progress_normal_mean,(6,-1))
                drought_season_vals_normal = NDVI_progress_normal_mean_reshape[1][global_drought_season_index_dict[season]]
                drought_season_vals_normal_mean = np.nanmean(drought_season_vals_normal)

                # delta = drought_season_vals_hot_mean - drought_season_vals_normal_mean

                # spatial_dict[pix] = delta
                spatial_dict_hot[pix] = drought_season_vals_hot_mean
                spatial_dict_normal[pix] = drought_season_vals_normal_mean
            arr_hot = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_hot)
            outf_hot = join(outdir, f'hot_{season}.tif')
            DIC_and_TIF().arr_to_tif(arr_hot, outf_hot)

            arr_normal = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_normal)
            outf_normal = join(outdir, f'normal_{season}.tif')
            DIC_and_TIF().arr_to_tif(arr_normal, outf_normal)

        pass

    def season_excerbation_alleviation_ratio_statistic_CSIF(self):
        fdir = join(self.this_class_tif, 'season_excerbation_alleviation_ratio_CSIF')
        outdir = join(self.this_class_arr, 'season_excerbation_alleviation_ratio_statistic_CSIF')
        drt_list = ['hot', 'normal']
        T.mk_dir(outdir)
        spatial_dict_all = {}
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir, f)
            # arr = DIC_and_TIF().spatial_tif_to_arr(fpath)
            # plt.imshow(arr, cmap='RdBu', vmin=-0.2, vmax=0.2, interpolation='nearest')
            # plt.colorbar()
            # plt.show()
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dict_mode = {}
            for pix in spatial_dict:
                val = spatial_dict[pix]
                if np.isnan(val):
                    continue

                if val > 5:
                    mode = 'alleviation'
                elif val < -5:
                    mode = 'excerbation'
                else:
                    mode = 'normal'
                    # raise ValueError
                spatial_dict_mode[pix] = mode
            key = f.replace('.tif', '')
            spatial_dict_all[key] = spatial_dict_mode
        df = T.spatial_dics_to_df(spatial_dict_all)
        df = Dataframe_func(df).df
        # T.print_head_n(df);exit()
        T.print_head_n(df)
        ratio_dict = {}
        ELI_class_list = global_ELI_class_list
        flag = 0
        for drt in drt_list:
            for season in global_drought_season_list:
                col_name = f'{drt}_{season}'
                df_season = df.dropna(subset=[col_name])
                for ELI_class in ELI_class_list:
                    df_ELI = df_season[df_season['ELI_class'] == ELI_class]
                    mode_list = T.get_df_unique_val_list(df_ELI, col_name)
                    # print(mode_list)
                    for mode in mode_list:
                        df_mode = df_ELI[df_ELI[col_name] == mode]
                        count = len(df_mode)
                        ratio = count / len(df_ELI)
                        # print(season, ELI_class, mode, count, f'{ratio:.3f}%')
                        ratio_dict[flag] = {
                            'season': season,
                            'drt': drt,
                            'ELI_class': ELI_class,
                            'mode': mode,
                            'count': count,
                            'ratio': f'{ratio*100:.3f}%'
                        }
                        flag += 1
        df_result = T.dic_to_df(ratio_dict, 'flag')
        T.print_head_n(df_result)
        outf = join(outdir, 'ratio')
        T.df_to_excel(df_result, outf)

        pass

    def plot_season_excerbation_alleviation_ratio_CSIF(self):
        fpath = join(self.this_class_arr, 'season_excerbation_alleviation_ratio_statistic_CSIF', 'ratio.xlsx')
        outdir = join(self.this_class_png, 'plot_season_excerbation_alleviation_ratio_CSIF')
        T.mk_dir(outdir)
        df = pd.read_excel(fpath, index_col=0)
        # T.print_head_n(df);exit()
        drt_list = ['hot', 'normal']
        ELI_class_list = global_ELI_class_list
        drought_season_list = global_drought_season_list
        mode_list = ['alleviation', 'excerbation']

        for ELI in ELI_class_list:
            plt.figure(figsize=(5, 5))
            df_ELI = df[df['ELI_class'] == ELI]
            for season in drought_season_list:
                df_season = df_ELI[df_ELI['season'] == season]
                for drt in drt_list[::-1]:
                    df_drt = df_season[df_season['drt'] == drt]
                    for mode in mode_list:
                        df_mode = df_drt[df_drt['mode'] == mode]
                        print(df_mode)
                        if len(df_mode) != 1:
                            raise
                        ratio_str = df_mode['ratio'].tolist()[0]
                        ratio = ratio_str.split('%')[0]
                        ratio = float(ratio)
                        color = 'blue'
                        if mode == 'excerbation':
                            ratio = -ratio
                            color = 'red'
                        plt.bar(f'{season} {drt}', ratio, color=color)
                        # key = f'{season} {ELI} {drt} {mode}'
                        # bar_dict[key] = {'ratio':ratio,'color':color}
            # pprint(bar_dict)
            plt.title(f'{ELI}')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.ylim(-95, 40)
            outf = join(outdir, f'{ELI}.pdf')
            plt.savefig(outf, dpi=300)
            plt.close()
            # plt.show()

    def season_excerbation_alleviation_ratio_tif_GPP_NIRv(self,df):
        # T.print_head_n(df)
        outdir = join(self.this_class_tif, 'season_excerbation_alleviation_ratio_GPP_NIRv')
        T.mk_dir(outdir)
        global_drought_season_index_dict = {
            'spring': [0,1],
            'summer': [2,3],
            'autumn': [4,5],
        }
        drought_season_list = global_drought_season_list
        ELI_class_list = global_ELI_class_list[::-1]
        alleviation_list = []
        excerbation_list = []
        progress_col_name = 'GPP_NIRv-percentage_progress'

        for season in drought_season_list:
            df_season = df[df['drought_season'] == season]
            df_pix_dict = T.df_groupby(df_season, 'pix')
            spatial_dict_normal = {}
            spatial_dict_hot = {}
            for pix in tqdm(df_pix_dict):
                df_pix = df_pix_dict[pix]
                df_hot = df_pix[df_pix['drought_type'] == 'hot-drought']
                df_normal = df_pix[df_pix['drought_type'] == 'normal-drought']
                NDVI_progress_hot = df_hot[progress_col_name].tolist()
                NDVI_progress_normal = df_normal[progress_col_name].tolist()
                if len(NDVI_progress_hot) == 0 or len(NDVI_progress_normal) == 0:
                    continue
                NDVI_progress_hot_mean = np.nanmean(NDVI_progress_hot,axis=0)
                # print(NDVI_progress_hot_mean)
                # print(type(NDVI_progress_hot_mean))
                if type(NDVI_progress_hot_mean) == np.float64:
                    continue
                # print(NDVI_progress_hot_mean)
                NDVI_progress_hot_mean_reshape = np.reshape(NDVI_progress_hot_mean,(6,-1))
                drought_season_vals_hot = NDVI_progress_hot_mean_reshape[1][global_drought_season_index_dict[season]]
                drought_season_vals_hot_mean = np.nanmean(drought_season_vals_hot)

                NDVI_progress_normal_mean = np.nanmean(NDVI_progress_normal,axis=0)
                NDVI_progress_normal_mean_reshape = np.reshape(NDVI_progress_normal_mean,(6,-1))
                drought_season_vals_normal = NDVI_progress_normal_mean_reshape[1][global_drought_season_index_dict[season]]
                drought_season_vals_normal_mean = np.nanmean(drought_season_vals_normal)

                # delta = drought_season_vals_hot_mean - drought_season_vals_normal_mean

                # spatial_dict[pix] = delta
                spatial_dict_hot[pix] = drought_season_vals_hot_mean
                spatial_dict_normal[pix] = drought_season_vals_normal_mean
            arr_hot = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_hot)
            outf_hot = join(outdir, f'hot_{season}.tif')
            DIC_and_TIF().arr_to_tif(arr_hot, outf_hot)

            arr_normal = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_normal)
            outf_normal = join(outdir, f'normal_{season}.tif')
            DIC_and_TIF().arr_to_tif(arr_normal, outf_normal)

        pass

    def season_excerbation_alleviation_ratio_statistic_GPP_NIRv(self):
        fdir = join(self.this_class_tif, 'season_excerbation_alleviation_ratio_GPP_NIRv')
        outdir = join(self.this_class_arr, 'season_excerbation_alleviation_ratio_statistic_GPP_NIRv')
        drt_list = ['hot', 'normal']
        T.mk_dir(outdir)
        spatial_dict_all = {}
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir, f)
            # arr = DIC_and_TIF().spatial_tif_to_arr(fpath)
            # plt.imshow(arr, cmap='RdBu', vmin=-0.2, vmax=0.2, interpolation='nearest')
            # plt.colorbar()
            # plt.show()
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dict_mode = {}
            for pix in spatial_dict:
                val = spatial_dict[pix]
                if np.isnan(val):
                    continue

                if val > 5:
                    mode = 'alleviation'
                elif val < -5:
                    mode = 'excerbation'
                else:
                    mode = 'normal'
                    # raise ValueError
                spatial_dict_mode[pix] = mode
            key = f.replace('.tif', '')
            spatial_dict_all[key] = spatial_dict_mode
        df = T.spatial_dics_to_df(spatial_dict_all)
        df = Dataframe_func(df).df
        # T.print_head_n(df);exit()
        T.print_head_n(df)
        ratio_dict = {}
        ELI_class_list = global_ELI_class_list
        flag = 0
        for drt in drt_list:
            for season in global_drought_season_list:
                col_name = f'{drt}_{season}'
                df_season = df.dropna(subset=[col_name])
                for ELI_class in ELI_class_list:
                    df_ELI = df_season[df_season['ELI_class'] == ELI_class]
                    mode_list = T.get_df_unique_val_list(df_ELI, col_name)
                    # print(mode_list)
                    for mode in mode_list:
                        df_mode = df_ELI[df_ELI[col_name] == mode]
                        count = len(df_mode)
                        ratio = count / len(df_ELI)
                        # print(season, ELI_class, mode, count, f'{ratio:.3f}%')
                        ratio_dict[flag] = {
                            'season': season,
                            'drt': drt,
                            'ELI_class': ELI_class,
                            'mode': mode,
                            'count': count,
                            'ratio': f'{ratio*100:.3f}%'
                        }
                        flag += 1
        df_result = T.dic_to_df(ratio_dict, 'flag')
        T.print_head_n(df_result)
        outf = join(outdir, 'ratio')
        T.df_to_excel(df_result, outf)

        pass

    def plot_season_excerbation_alleviation_ratio_GPP_NIRv(self):
        fpath = join(self.this_class_arr, 'season_excerbation_alleviation_ratio_statistic_GPP_NIRv', 'ratio.xlsx')
        outdir = join(self.this_class_png, 'plot_season_excerbation_alleviation_ratio_GPP_NIRv')
        T.mk_dir(outdir)
        df = pd.read_excel(fpath, index_col=0)
        # T.print_head_n(df);exit()
        drt_list = ['hot', 'normal']
        ELI_class_list = global_ELI_class_list
        drought_season_list = global_drought_season_list
        mode_list = ['alleviation', 'excerbation']

        for ELI in ELI_class_list:
            plt.figure(figsize=(5, 5))
            df_ELI = df[df['ELI_class'] == ELI]
            for season in drought_season_list:
                df_season = df_ELI[df_ELI['season'] == season]
                for drt in drt_list[::-1]:
                    df_drt = df_season[df_season['drt'] == drt]
                    for mode in mode_list:
                        df_mode = df_drt[df_drt['mode'] == mode]
                        print(df_mode)
                        if len(df_mode) != 1:
                            raise
                        ratio_str = df_mode['ratio'].tolist()[0]
                        ratio = ratio_str.split('%')[0]
                        ratio = float(ratio)
                        color = 'blue'
                        if mode == 'excerbation':
                            ratio = -ratio
                            color = 'red'
                        plt.bar(f'{season} {drt}', ratio, color=color)
                        # key = f'{season} {ELI} {drt} {mode}'
                        # bar_dict[key] = {'ratio':ratio,'color':color}
            # pprint(bar_dict)
            plt.title(f'{ELI}')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.ylim(-95, 40)
            outf = join(outdir, f'{ELI}.pdf')
            plt.savefig(outf, dpi=300)
            plt.close()
            # plt.show()

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

    def GEZ_statistic(self):
        outdir = join(self.this_class_png, 'GEZ_statistic')
        T.mk_dir(outdir)
        fpath = join(self.this_class_tif,'delta/delta.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
        df = T.spatial_dics_to_df({'delta':spatial_dict})
        df = Dataframe_func(df).df
        T.print_head_n(df)
        gez_list = T.get_df_unique_val_list(df, 'GEZ')
        gez_list = list(gez_list)
        gez_list.remove('Water')
        print(gez_list)
        count_list = []
        count_list_humid = []
        count_list_arid = []
        delta_list = []
        # delta_max_list = []
        delta_25_list = []
        bar_delta_list = []
        for gez in gez_list:
            df_gez = df[df['GEZ'] == gez]
            df_gez_humid = df_gez[df_gez['AI_class'] == 'Humid']
            df_gez_arid = df_gez[df_gez['AI_class'] == 'Arid']
            # T.print_head_n(df_gez);exit()
            delta = df_gez['delta'].tolist()
            count_list.append(len(df_gez))
            count_list_humid.append(len(df_gez_humid))
            count_list_arid.append(len(df_gez_arid))
            delta_mean = np.nanmean(delta)
            delta_25_percentile = np.percentile(delta,25)
            delta_75_percentile = np.percentile(delta,75)
            # delta_max = np.nanmax(delta)
            # delta_min = np.nanmin(delta)
            # count = len(df_gez)
            # count_list.append(count)
            delta_list.append(delta_mean)
            bar_delta = delta_75_percentile - delta_25_percentile
            bar_delta_list.append(bar_delta)
            delta_25_list.append(delta_25_percentile)
        # plt.scatter(gez_list,delta_max_list,label='max')
        # plt.scatter(gez_list,delta_min_list,label='min')
        plt.bar(gez_list,bar_delta_list,bottom=delta_25_list)
        plt.scatter(gez_list,delta_list,label='mean')
        # plt.legend()
        plt.xticks(rotation=90)
        plt.tight_layout()

        outf = join(outdir, 'delta_bar.pdf')
        # plt.savefig(outf)
        plt.close()

        plt.figure()
        # plt.bar(gez_list,count_list)
        # print(count_list_humid);exit()
        plt.bar(gez_list,count_list_humid,color='b')
        plt.bar(gez_list,count_list_arid,bottom=count_list_humid,color='r')
        plt.xticks(rotation=90)
        plt.tight_layout()
        outf = join(outdir,'count.pdf')
        plt.savefig(outf)
        plt.close()
        # plt.show()
        pass

    def high_latitude_hot_drought(self):
        df = self.__gen_df_init()
        df_hotdrought_50N = df[df['lat'] >= 60]
        df_hotdrought = df_hotdrought_50N[df_hotdrought_50N['drought_type'] == 'hot-drought']
        df_hotdrought_increase_NDVI = df_hotdrought[df_hotdrought['rt'] > 1]
        ratio = len(df_hotdrought_increase_NDVI) / len(df_hotdrought)
        df_pix_dict = T.df_groupby(df_hotdrought, 'pix')
        spatial_dict = {}
        for pix in df_pix_dict:
            df_pix = df_pix_dict[pix]
            val = df_pix['rt'].tolist()
            spatial_dict[pix] = np.nanmean(val)
        print(ratio)

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        plt.imshow(arr,interpolation='nearest',vmin=0.9,vmax=1.1,cmap='RdBu',aspect='auto')
        plt.colorbar()
        plt.show()

        pass

    def add_VPD_anomaly_process(self,df):
        # df = Load_dataframe()
        NDVI_spatial_dict,data_name,valid_range = Load_Data().VPD_anomaly()
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

    def add_radiation_process(self,df):
        # df = Load_dataframe()
        NDVI_spatial_dict,data_name,valid_range = Load_Data().VPD_anomaly()
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

    def add_VPD_origin_process(self,df):
        # df = Load_dataframe()
        NDVI_spatial_dict,data_name,valid_range = Load_Data().VPD_origin()
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

    def add_NDVI_percentage_process(self,df):
        # df = Load_dataframe()
        NDVI_spatial_dict,data_name,valid_range = Load_Data().NDVI_percentage()
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
            NDVI[NDVI>valid_range[1]] = np.nan
            NDVI[NDVI<valid_range[0]] = np.nan
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

    def add_CSIF_percentage_process(self,df):
        # df = Load_dataframe()
        NDVI_spatial_dict,data_name,valid_range = Load_Data().CSIF_percentage()
        # print(data_name)
        # exit()
        year_list = year_range_str_to_list(global_VIs_year_range_dict['CSIF'])
        gs = global_gs
        NDVI_list_all = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            NDVI = NDVI_spatial_dict[pix]
            NDVI = np.array(NDVI,dtype=float)
            NDVI[NDVI>valid_range[1]] = np.nan
            NDVI[NDVI<valid_range[0]] = np.nan
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

    def add_CSIF_anomaly_process(self,df):
        # df = Load_dataframe()
        NDVI_spatial_dict,data_name,valid_range = Load_Data().CSIF_anomaly_detrend()
        # print(data_name)
        # exit()
        year_list_str = global_VIs_year_range_dict['CSIF']
        year_list = year_range_str_to_list(year_list_str)
        # print(year_list);exit()
        gs = global_gs
        NDVI_list_all = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            NDVI = NDVI_spatial_dict[pix]
            NDVI = np.array(NDVI,dtype=float)
            NDVI[NDVI>valid_range[1]] = np.nan
            NDVI[NDVI<valid_range[0]] = np.nan
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

    def add_GPP_NIRv_percentage_process(self,df):
        # df = Load_dataframe()
        NDVI_spatial_dict,data_name,valid_range = Load_Data().GPP_NIRv_percentage()
        # print(data_name)
        # exit()
        year_list = year_range_str_to_list(global_VIs_year_range_dict['GPP_NIRv'])
        gs = global_gs
        NDVI_list_all = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            if not pix in NDVI_spatial_dict:
                NDVI_list_all.append(np.nan)
                continue
            NDVI = NDVI_spatial_dict[pix]
            NDVI = np.array(NDVI,dtype=float)
            NDVI[NDVI>valid_range[1]] = np.nan
            NDVI[NDVI<valid_range[0]] = np.nan
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

    def add_GPP_NIRv_anomaly_process(self,df):
        # df = Load_dataframe()
        NDVI_spatial_dict,data_name,valid_range = Load_Data().GPP_NIRv_anomaly()
        # print(data_name)
        # exit()
        year_list_str = global_VIs_year_range_dict['GPP_NIRv']
        year_list = year_range_str_to_list(year_list_str)
        # print(year_list);exit()
        gs = global_gs
        NDVI_list_all = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            if not pix in NDVI_spatial_dict:
                NDVI_list_all.append(np.nan)
                continue
            NDVI = NDVI_spatial_dict[pix]
            NDVI = np.array(NDVI,dtype=float)
            NDVI[NDVI>valid_range[1]] = np.nan
            NDVI[NDVI<valid_range[0]] = np.nan
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

    def add_NIRv_percentage_process(self,df):
        # df = Load_dataframe()
        NDVI_spatial_dict,data_name,valid_range = Load_Data().NIRv_percentage()
        # print(data_name)
        # exit()
        year_list = year_range_str_to_list(global_VIs_year_range_dict['NIRv'])
        gs = global_gs
        NDVI_list_all = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            if not pix in NDVI_spatial_dict:
                NDVI_list_all.append(np.nan)
                continue
            NDVI = NDVI_spatial_dict[pix]
            NDVI = np.array(NDVI,dtype=float)
            NDVI[NDVI>valid_range[1]] = np.nan
            NDVI[NDVI<valid_range[0]] = np.nan
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

    def add_NIRv_anomaly_process(self,df):
        # df = Load_dataframe()
        NDVI_spatial_dict,data_name,valid_range = Load_Data().NIRv_anomaly()
        # print(data_name)
        # exit()
        year_list_str = global_VIs_year_range_dict['NIRv']
        year_list = year_range_str_to_list(year_list_str)
        # print(year_list);exit()
        gs = global_gs
        NDVI_list_all = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            if not pix in NDVI_spatial_dict:
                NDVI_list_all.append(np.nan)
                continue
            NDVI = NDVI_spatial_dict[pix]
            NDVI = np.array(NDVI,dtype=float)
            NDVI[NDVI>valid_range[1]] = np.nan
            NDVI[NDVI<valid_range[0]] = np.nan
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


    def VPD_delta_tif(self,df):
        outdir = join(self.this_class_tif, 'VPD_delta')
        T.mk_dir(outdir)
        drought_season_list = global_drought_season_list
        # print(df_season)
        pix_list = T.get_df_unique_val_list(df, 'pix')
        # print(len(pix_list))
        drought_type_list = global_drought_type_list
        col_name = 'VPD-origin_progress'
        # col_name = 'VPD-anomaly_progress'
        # drought_year_list = range(1, 5)
        drought_year_list = [1]
        for drought_year_i in drought_year_list:
            df_group_dict = T.df_groupby(df, 'pix')
            spatial_dict = {}
            for pix in tqdm(pix_list):
                df_pix = df_group_dict[pix]
                df_hot = df_pix[df_pix['drought_type'] == 'hot-drought']
                df_normal = df_pix[df_pix['drought_type'] == 'normal-drought']
                if len(df_hot) == 0 or len(df_normal) == 0:
                    continue
                NDVI_progress_hot = df_hot[col_name].tolist()
                NDVI_progress_normal = df_normal[col_name].tolist()
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
            outf = join(outdir, f'{col_name}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict, outf)

    def VPD_alleviation_excerbation(self):
        VPD_fpath = join(self.this_class_tif, 'VPD_delta', 'VPD_delta.tif')
        # VPD_fpath = join(self.this_class_tif, 'VPD_delta', 'VPD-origin_progress.tif')
        delta_fpath = join(self.this_class_tif, 'delta', 'delta.tif')

        delta_dict = DIC_and_TIF().spatial_tif_to_dic(delta_fpath)
        VPD_dict = DIC_and_TIF().spatial_tif_to_dic(VPD_fpath)

        spatial_dicts = {
            'VPD': VPD_dict,
            'delta': delta_dict
        }

        df = T.spatial_dics_to_df(spatial_dicts)
        df = Dataframe_func(df).df
        T.print_head_n(df, 10)

        # vpd_range = (-0.1, 0.5)
        vpd_range = (-2, 2)
        ELI_class_list = global_ELI_class_list
        VPD_bins = np.linspace(vpd_range[0],vpd_range[1],50)
        delta_bins = np.linspace(-2,2,40)
        for ELI in ELI_class_list:
            plt.figure()

            df_ELI = df[df['ELI_class'] == ELI]
            # df_ELI = df
            # VPD_vals = df_ELI['VPD'].tolist()
            # delta_vals = df_ELI['delta'].tolist()
            VPD_vals = df_ELI['VPD'].tolist()
            # plt.hist(VPD_vals, bins=VPD_bins, alpha=0.5, label=ELI)
            # plt.twinx()
            x_,y_ = Plot().plot_hist_smooth(VPD_vals, alpha=0,bins=100,range=vpd_range)
            df_group, bins_list_str = T.df_bin(df_ELI,'VPD',VPD_bins)
            x_list = []
            y_list = []
            err_list = []
            for name,df_group_i in df_group:
                x = name[0].left
                vals = df_group_i['delta'].tolist()
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                x_list.append(x)
                y_list.append(mean)
                err_list.append(std)
            y_list = np.array(y_list)
            err_list = np.array(err_list)
            plt.title(ELI)
            plt.plot(x_,y_)

            plt.twinx()
            plt.hlines(0,vpd_range[0],vpd_range[1],linestyles='dashed',colors='k')
            plt.plot(x_list,y_list, label=ELI)
            plt.fill_between(x_list,y_list-err_list,y_list+err_list,alpha=0.5)
            plt.legend()
        plt.show()

        pass


    def VPD_NDVI(self,df):

        # VPD_name = 'VPD-origin_progress'
        VPD_name = 'VPD-anomaly_progress'
        NDVI_vals_list = []
        VPD_vals_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            VPD_vals = row[VPD_name].tolist()
            VPD_reshape = np.array(VPD_vals).reshape(-1, 6)
            VPD_drought_year = VPD_reshape[1]

            NDVI_vals = row['NDVI_progress'].tolist()
            NDVI_reshape = np.array(NDVI_vals).reshape(-1, 6)
            NDVI_drought_year = NDVI_reshape[1]

            for VPD in VPD_drought_year:
                VPD_vals_list.append(VPD)
            for NDVI in NDVI_drought_year:
                NDVI_vals_list.append(NDVI)
        df_new = pd.DataFrame({'VPD':VPD_vals_list,'NDVI':NDVI_vals_list})
        # plt.hist(VPD_vals_list,bins=40)
        # plt.twinx()
        # T.print_head_n(df_new,10)
        # VPD_bins = np.linspace(0, 4, 40)
        VPD_bins = np.linspace(-3, 3, 40)
        # VPD_vals = df_new['VPD'].tolist()
        df_group, bins_list_str = T.df_bin(df_new,'VPD',VPD_bins)

        x_list = []
        y_list = []
        err_list = []
        for name,df_group_i in df_group:
            x = name[0].left
            vals = df_group_i['NDVI'].tolist()
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            x_list.append(x)
            y_list.append(mean)
            err_list.append(std)
        y_list = np.array(y_list)
        err_list = np.array(err_list)
        plt.plot(x_list,y_list)
        plt.fill_between(x_list,y_list-err_list,y_list+err_list,alpha=0.5)
        # plt.legend()
        plt.show()
        pass

    def Drought_year_spatial_tif(self,df):
        col_name = 'NIRv-anomaly_progress'

        outdir = join(self.this_class_tif,'Drought_year_spatial_tif',col_name)
        T.mk_dir(outdir,force=True)
        # col_name = 'NDVI_progress'
        drought_type_list = global_drought_type_list
        for drt in drought_type_list:
            df_drt = df[df['drought_type']==drt]
            df_pix = T.df_groupby(df_drt,'pix')
            spatial_dict = {}
            for pix in tqdm(df_pix,desc=drt):
                df_pix_i = df_pix[pix]
                NDVI_progress = df_pix_i[col_name].tolist()
                mean_progress = np.nanmean(NDVI_progress, axis=0)
                mean_progress_reshape = np.array(mean_progress).reshape(-1, 6)
                mean_progress_reshape_drought_year = mean_progress_reshape[1]
                mean_drought_year_NDVI = np.nanmean(mean_progress_reshape_drought_year)
                spatial_dict[pix] = mean_drought_year_NDVI
            outf = join(outdir,'{}.tif'.format(drt))
            DIC_and_TIF().pix_dic_to_tif(spatial_dict,outf)

    def Drought_year_percentage_spatial_tif(self,df):

        col_name = 'NIRv-percentage_progress'
        outdir = join(self.this_class_tif,'Drought_year_percentage_spatial_tif',col_name)
        # NDVI_data_dict = Load_Data().NDVI_origin()
        T.mk_dir(outdir,force=True)
        drought_type_list = global_drought_type_list
        for drt in drought_type_list:
            df_drt = df[df['drought_type']==drt]
            df_pix = T.df_groupby(df_drt,'pix')
            spatial_dict = {}
            for pix in tqdm(df_pix,desc=drt):
                df_pix_i = df_pix[pix]
                NDVI_progress = df_pix_i[col_name].tolist()
                # print(NDVI_progress)
                NDVI_progress_flatten = np.array(NDVI_progress).flatten()
                if T.is_all_nan(NDVI_progress_flatten):
                    continue
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

    def Drought_year_percentage_sig_spatial_tif(self):
        col_name = 'NIRv-percentage_progress'
        fdir = join(self.this_class_tif,'Drought_year_percentage_spatial_tif',col_name)
        outdir = join(self.this_class_tif,'Drought_year_percentage_sig_spatial_tif',col_name)
        T.mk_dir(outdir,force=True)
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            df = T.spatial_dics_to_df({'val':spatial_dict})

            sig_list = []
            for i,row in df.iterrows():
                val = row['val']
                if val > 5 or val < -5:
                    sig_list.append(0.0001)
                else:
                    sig_list.append(1)
            df['sig'] = sig_list
            T.print_head_n(df)
            spatial_dict_sig = T.df_to_spatial_dic(df,'sig')
            # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_sig)
            DIC_and_TIF().pix_dic_to_tif(spatial_dict_sig,outpath)

    def plot_Drought_year_spatial_tif(self):
        product = 'NIRv'
        col_name_anomaly = f'{product}-anomaly_progress'
        col_name_percentage = f'{product}-percentage_progress'
        fdir = join(self.this_class_tif,'Drought_year_spatial_tif',col_name_anomaly)
        fdir_sig = join(self.this_class_tif,'Drought_year_percentage_sig_spatial_tif',col_name_percentage)

        outdir = join(self.this_class_png,'Drought_year_spatial_tif',product)
        T.mk_dir(outdir,force=True)
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


class Dynamic_gs_analysis:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Dynamic_gs_analysis', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'Dataframe.df')
        # print(self.this_class_arr);exit()
        pass

    def run(self):
        # self.copy_df()
        df = self.__gen_df_init()
        # self.add_T_anomaly_to_df(df)
        # self.add_NDVI_anomaly_to_df(df)
        # self.add_NDVI_anomaly_drought_year_to_df(df)
        # self.add_NDVI_percentage_to_df(df)
        # self.add_NDVI_percentage_drought_year_to_df(df)
        # self.add_NDVI_anomaly_whole_year_to_df(df)
        # self.add_early_peak_late_period_to_df(df)
        # df = self.add_drought_season_to_df(df)
        # df = self.add_AI_new_to_df(df)
        # df = self.add_AI_new_class_to_df(df)
        # df = self.add_eos(df)
        # T.print_head_n(df);exit()
        # T.save_df(df, self.dff)
        # T.df_to_excel(df, self.dff)

        # df = df.dropna(subset=['early_range'])

        # self.Figure1ab(df)
        # self.Figure1ab_percentage(df)
        # self.plot_Figure1ab()
        # self.Figure1c()
        # self.plot_Figure1c()
        # self.Figure1de()
        # self.Figure1f()
        # self.Figure2a()
        # self.Figure2b()
        # self.Figure3a(df)
        # self.Figure3a_count(df)
        # self.Figure3a_cdf()
        # self.Figure3b()
        # self.Figure4_values(df)
        # self.Figure4_values_std(df)
        # self.Figure4_values_boxplot(df)
        # self.season_excerbation_alleviation_ratio_tif(df)
        # self.season_excerbation_alleviation_ratio_statistic()
        # self.plot_season_excerbation_alleviation_ratio()
        # self.plot_drought_events_timeseries_different_season(df)
        # self.plot_spatial_trend_of_temperature_during_drought(df)
        # self.plot_temporal_trend_of_temperature_during_drought(df)
        # self.plot_NDVI_seasonal_time_series_during_drought(df)
        # self.Figure4_ratio(df)
        # self.Figure4_trajectory(df)
        # self.Figure_S6(df)
        # self.Figure_S6_1(df)
        # self.SOS_drought_timing_AI(df)
        # self.plot_SOS_NDVI_seasonal_time_series_during_drought(df)
        # self.plot_SOS_NDVI_seasonal_bar_during_drought(df)
        # self.plot_seasonal_drought_number_gradient(df)
        # self.plot_Temperature_vs_SOS(df)
        # self.print_early_peak_late_reduction(df)
        # self.plot_SOS_during_drought(df)

        # self.GS_length_during_drought(df)
        # self.plot_GS_length_during_drought()
        # self.GS_length_during_drought_vs_AI()
        # self.plot_AI_histogram(df)
        # self.GS_length_during_drought_advanced_delayed_SOS(df)
        # self.check_df(df)

        pass

    def Figure1ab(self,df):
        outdir = join(self.this_class_tif,'Figure1ab')
        T.mk_dir(outdir)
        NDVI_dict,_,_ = Load_Data().NDVI_anomaly_detrend()
        NDVI_drought_growing_season_mean_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            growing_season = row['growing_season']
            if type(growing_season) == float:
                NDVI_drought_growing_season_mean_list.append(np.nan)
                continue
            drought_year = row['drought_year']
            NDVI_vals = NDVI_dict[pix]
            NDVI_vals_reshape = np.array(NDVI_vals).reshape(-1,12)
            NDVI_vals_annual_dict = T.dict_zip(global_year_range_list,NDVI_vals_reshape)
            NDVI_drought_year = NDVI_vals_annual_dict[drought_year]
            NDVI_drought_growing_season = NDVI_drought_year[growing_season-1]
            NDVI_drought_growing_season_mean = np.nanmean(NDVI_drought_growing_season)
            NDVI_drought_growing_season_mean_list.append(NDVI_drought_growing_season_mean)
        df['NDVI_anomaly_drought_growing_season'] = NDVI_drought_growing_season_mean_list

        drt_list = ['normal-drought','hot-drought']
        for drt in drt_list:
            df_drt = df[df['drought_type'] == drt]
            df_drought_spatial_dict = T.df_groupby(df_drt,'pix')
            spatial_dict_NDVI = {}
            for pix in tqdm(df_drought_spatial_dict,desc=drt):
                df_i = df_drought_spatial_dict[pix]
                NDVI_anomaly_drought_growing_season = df_i['NDVI_anomaly_drought_growing_season'].tolist()
                NDVI_anomaly_drought_growing_season_mean = np.nanmean(NDVI_anomaly_drought_growing_season)
                spatial_dict_NDVI[pix] = NDVI_anomaly_drought_growing_season_mean
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_NDVI)
            outf = join(outdir,f'{drt}.tif')
            DIC_and_TIF().arr_to_tif(arr,outf)
        T.open_path_and_file(outdir)
        pass

    def Figure1ab_percentage(self,df):
        outdir = join(self.this_class_tif,'Figure1ab_percentage')
        NDVI_dict, _, _ = Load_Data().NDVI_percentage()
        NDVI_drought_growing_season_mean_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            growing_season = row['growing_season']
            if type(growing_season) == float:
                NDVI_drought_growing_season_mean_list.append(np.nan)
                continue
            drought_year = row['drought_year']
            NDVI_vals = NDVI_dict[pix]
            NDVI_vals_reshape = np.array(NDVI_vals).reshape(-1, 12)
            NDVI_vals_annual_dict = T.dict_zip(global_year_range_list, NDVI_vals_reshape)
            NDVI_drought_year = NDVI_vals_annual_dict[drought_year]
            # plt.plot(NDVI_vals)
            # plt.show()
            NDVI_drought_growing_season = NDVI_drought_year[growing_season - 1]
            NDVI_drought_growing_season_mean = np.nanmean(NDVI_drought_growing_season)
            # NDVI_drought_growing_season_mean = abs(NDVI_drought_growing_season_mean)
            # print(NDVI_drought_growing_season_mean);exit()
            # if NDVI_drought_growing_season_mean > 5:
            #     sig_val = 0
            # else:
            #     sig_val = 1
            NDVI_drought_growing_season_mean_list.append(NDVI_drought_growing_season_mean)
        df['NDVI_anomaly_drought_growing_season'] = NDVI_drought_growing_season_mean_list

        drt_list = ['normal-drought', 'hot-drought']
        for drt in drt_list:
            df_drt = df[df['drought_type'] == drt]
            df_drought_spatial_dict = T.df_groupby(df_drt, 'pix')
            spatial_dict_NDVI_sig = {}
            spatial_dict_NDVI_values = {}
            for pix in tqdm(df_drought_spatial_dict, desc=drt):
                df_i = df_drought_spatial_dict[pix]
                NDVI_anomaly_drought_growing_season = df_i['NDVI_anomaly_drought_growing_season'].tolist()
                NDVI_anomaly_drought_growing_season_mean = np.nanmean(NDVI_anomaly_drought_growing_season)
                NDVI_anomaly_drought_growing_season_mean = abs(NDVI_anomaly_drought_growing_season_mean)
                if NDVI_anomaly_drought_growing_season_mean > 5:
                    sig_val = 0
                else:
                    sig_val = 1
                spatial_dict_NDVI_sig[pix] = sig_val
                spatial_dict_NDVI_values[pix] = NDVI_anomaly_drought_growing_season_mean
            arr_sig = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_NDVI_sig)
            arr_values = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_NDVI_values)
            outf_values = join(outdir, f'{drt}_values.tif')
            DIC_and_TIF().arr_to_tif(arr_values, outf_values)
            outf_sig = join(outdir, f'{drt}_sig.tif')
            DIC_and_TIF().arr_to_tif(arr_sig, outf_sig)
        # T.open_path_and_file(outdir)

    def Figure1c(self):
        fdir = join(self.this_class_tif,'Figure1ab')
        fdir_sig = join(self.this_class_tif,'Figure1ab_percentage')
        outdir = join(self.this_class_tif,'Figure1c')
        T.mk_dir(outdir)
        normal_drought_tif = join(fdir,'normal-drought.tif')
        hot_drought_tif = join(fdir,'hot-drought.tif')
        normal_drought_arr = DIC_and_TIF().spatial_tif_to_arr(normal_drought_tif)
        hot_drought_arr = DIC_and_TIF().spatial_tif_to_arr(hot_drought_tif)
        arr = hot_drought_arr - normal_drought_arr
        outf = join(outdir,'NDVI_anomaly_drought_season.tif')
        DIC_and_TIF().arr_to_tif(arr,outf)

        normal_drought_sig_tif = join(fdir_sig,'normal-drought_values.tif')
        hot_drought_sig_tif = join(fdir_sig,'hot-drought_values.tif')
        normal_drought_sig_arr = DIC_and_TIF().spatial_tif_to_arr(normal_drought_sig_tif)
        hot_drought_sig_arr = DIC_and_TIF().spatial_tif_to_arr(hot_drought_sig_tif)
        arr_sig = hot_drought_sig_arr - normal_drought_sig_arr
        arr_sig = abs(arr_sig)
        arr_copy = arr_sig.copy() * np.nan
        arr_copy[arr_sig>=5] = 0
        arr_copy[arr_sig<5] = 1
        # plt.imshow(arr_copy,interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        outf_sig = join(outdir,'NDVI_anomaly_drought_season_sig.tif')
        DIC_and_TIF().arr_to_tif(arr_copy,outf_sig)
        T.open_path_and_file(outdir)
        pass

    def plot_Figure1ab(self):
        fdir = join(self.this_class_tif,'Figure1ab')
        fdir_sig = join(self.this_class_tif,'Figure1ab_percentage')
        outdir = join(self.this_class_png,'Figure1ab')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            fpath_sig = join(fdir_sig,f.replace('.tif','_sig.tif'))
            outf = join(outdir,f.replace('.tif','.png'))
            m,ret = Plot().plot_ortho(fpath,vmin=-1,vmax=1,cmap='RdBu')
            Plot().plot_ortho_significance_scatter(m, fpath_sig, temp_root,sig_level=0.05)
            # m,ret = Plot().plot_ortho_significance_scatter(fpath,vmin=-1,vmax=1,cmap='RdBu')
            plt.savefig(outf,dpi=900)
            # plt.title(f.replace('.tif',''))
            # plt.show()
            plt.close()
        T.open_path_and_file(outdir)

    def plot_Figure1c(self):
        fdir = join(self.this_class_tif,'Figure1c')
        outdir = join(self.this_class_png,'Figure1c')
        T.mk_dir(outdir)
        f = 'NDVI_anomaly_drought_season.tif'
        fpath = join(fdir,f)
        fpath_sig = join(fdir,f.replace('.tif','_sig.tif'))
        outf = join(outdir,f.replace('.tif','.png'))
        m,ret = Plot().plot_ortho(fpath,vmin=-1,vmax=1,cmap='Spectral')
        Plot().plot_ortho_significance_scatter(m, fpath_sig, temp_root,sig_level=0.05)
        plt.savefig(outf,dpi=900)
        # plt.title(f.replace('.tif',''))
        # plt.show()
        plt.close()
        T.open_path_and_file(outdir)

    def Figure1de(self):
        fdir = join(self.this_class_tif,'Figure1ab')
        fdir_percentage = join(self.this_class_tif,'Figure1ab_percentage')
        outdir = join(self.this_class_png,'Figure1de')
        T.mk_dir(outdir)
        # fpath_normal_drought = join(fdir,'normal-drought.tif')
        # fpath_hot_drought = join(fdir,'hot-drought.tif')
        # fpath_normal_drought_sig = join(fdir_sig,'normal-drought_values.tif')
        # fpath_hot_drought_sig = join(fdir_sig,'hot-drought_values.tif')
        path_normal_drought = 'normal-drought.tif'
        path_hot_drought = 'hot-drought.tif'
        path_normal_drought_percentage = 'normal-drought_values.tif'
        path_hot_drought_percentage = 'hot-drought_values.tif'

        all_spatial_dict = {}
        for f in [path_normal_drought,path_hot_drought,path_normal_drought_percentage,path_hot_drought_percentage]:
            if 'values' in f:
                fpath = join(fdir_percentage,f)
                key = f.replace('values.tif','percentage')
            else:
                fpath = join(fdir,f)
                key = f.replace('.tif','')
            # print(fpath)
            spatial_dict_i = DIC_and_TIF().spatial_tif_to_dic(fpath)
            all_spatial_dict[key] = spatial_dict_i
        df = T.spatial_dics_to_df(all_spatial_dict)
        df['hot-drought_percentage'] = abs(df['hot-drought_percentage'])
        df['normal-drought_percentage'] = abs(df['normal-drought_percentage'])
        df = Dataframe_func(df).df
        T.print_head_n(df)
        AI_class_list = global_AI_class_list
        plt.figure(figsize=(10,6))
        flag = 0
        for AI_class in AI_class_list:
            df_AI = df[df['AI_class'] == AI_class]
            drt_list = global_drought_type_list
            for drt in drt_list:
                df_AI_copy = df_AI.dropna(subset=[drt])

                df_negative = df_AI_copy[df_AI_copy[drt]<=0]
                df_positive = df_AI_copy[df_AI_copy[drt]>0]
                df_negative_sig = df_negative[df_negative[drt+'_percentage']>=5]
                df_positive_sig = df_positive[df_positive[drt+'_percentage']>=5]
                df_negative_ratio = len(df_negative)/len(df_AI_copy)
                df_positive_ratio = len(df_positive)/len(df_AI_copy)
                df_negative_ratio_sig = len(df_negative_sig)/len(df_AI_copy)
                df_positive_ratio_sig = len(df_positive_sig)/len(df_AI_copy)
                df_negative_ratio = -df_negative_ratio
                df_negative_ratio_sig = -df_negative_ratio_sig
                plt.bar([f'{drt}-{AI_class}'],[df_negative_ratio],width=0.5,alpha=0.5)
                plt.bar([f'{drt}-{AI_class}'],[df_positive_ratio],width=0.5,alpha=0.5)
                plt.bar([f'{drt}-{AI_class}'],[df_negative_ratio_sig],width=0.5,alpha=0.5)
                plt.bar([f'{drt}-{AI_class}'],[df_positive_ratio_sig],width=0.5,alpha=0.5)
                plt.text(flag, df_negative_ratio, f'{df_negative_ratio * 100:.2f}')
                plt.text(flag, df_positive_ratio, f'{df_positive_ratio * 100:.2f}')
                plt.text(flag, df_negative_ratio_sig, f'{df_negative_ratio_sig * 100:.2f}')
                plt.text(flag, df_positive_ratio_sig, f'{df_positive_ratio_sig * 100:.2f}')
                flag += 1
        plt.hlines(0,-1,4,linestyles='dashed')
        outf_percentage = join(outdir,'Figure1de_percentage.pdf')
        plt.savefig(outf_percentage,dpi=900)
        plt.close()

        plt.figure(figsize=(10,6))
        for drt in global_drought_type_list:
            for AI_class in AI_class_list:
                df_AI = df[df['AI_class'] == AI_class]
                vals = df_AI[drt].tolist()
                vals = np.array(vals)
                vals_mean = np.nanmean(vals)
                vals = vals[~np.isnan(vals)]
                err, up, bottom,err = self.mean_confidence_interval(vals)
                label = f'{drt}-{AI_class}'
                plt.bar([label],[vals_mean],yerr=[err],width=0.5,alpha=0.5)
                plt.hlines(0,-1,4,linestyles='dashed')
        outf = join(outdir,'Figure1de_values.pdf')
        plt.savefig(outf,dpi=900)
        plt.close()
        T.open_path_and_file(outdir)
        pass

    def Figure1f(self):
        outdir = join(self.this_class_png,'Figure1f')
        T.mk_dir(outdir)
        fdir = join(self.this_class_tif,'Figure1c')
        fpath_value = join(fdir,'NDVI_anomaly_drought_season.tif')
        fpath_sig = join(fdir,'NDVI_anomaly_drought_season_sig.tif')
        spatial_dict_value = DIC_and_TIF().spatial_tif_to_dic(fpath_value)
        spatial_dict_sig = DIC_and_TIF().spatial_tif_to_dic(fpath_sig)

        all_spatial_dict = {}
        all_spatial_dict['value'] = spatial_dict_value
        all_spatial_dict['percentage'] = spatial_dict_sig
        df = T.spatial_dics_to_df(all_spatial_dict)
        df = Dataframe_func(df).df
        T.print_head_n(df)
        plt.figure(figsize=(10, 6))
        for AI_class in global_AI_class_list:
            df_AI = df[df['AI_class'] == AI_class]
            plt.bar(AI_class,df_AI['value'].mean(),yerr=df_AI['value'].sem(),width=0.5,alpha=0.5)
        plt.hlines(0,-1,2,linestyles='dashed')
        outf1 = join(outdir,'Figure1f_value.pdf')
        plt.savefig(outf1)
        plt.close()

        plt.figure(figsize=(10, 6))
        flag = 0
        for AI_class in global_AI_class_list:
            df_AI = df[df['AI_class'] == AI_class]
            df_AI_negative = df_AI[df_AI['value'] < 0]
            df_AI_positive = df_AI[df_AI['value'] > 0]
            df_AI_negative_sig = df_AI_negative[df_AI_negative['percentage'] == 0]
            df_AI_positive_sig = df_AI_positive[df_AI_positive['percentage'] == 0]

            negative_ratio = len(df_AI_negative)/len(df_AI)
            positive_ratio = len(df_AI_positive)/len(df_AI)
            negative_ratio_sig = len(df_AI_negative_sig)/len(df_AI)
            positive_ratio_sig = len(df_AI_positive_sig)/len(df_AI)
            negative_ratio = -negative_ratio
            negative_ratio_sig = -negative_ratio_sig
            plt.bar([f'{AI_class}'],[negative_ratio],width=0.5,alpha=0.5)
            plt.bar([f'{AI_class}'],[positive_ratio],width=0.5,alpha=0.5)
            plt.bar([f'{AI_class}'],[negative_ratio_sig],width=0.5,alpha=0.5)
            plt.bar([f'{AI_class}'],[positive_ratio_sig],width=0.5,alpha=0.5)
            plt.text(flag,negative_ratio,f'{negative_ratio*100:.2f}')
            plt.text(flag,positive_ratio,f'{positive_ratio*100:.2f}')
            plt.text(flag,negative_ratio_sig,f'{negative_ratio_sig*100:.2f}')
            plt.text(flag,positive_ratio_sig,f'{positive_ratio_sig*100:.2f}')
            flag += 1

        plt.hlines(0,-1,2,linestyles='dashed')
        outf2 = join(outdir,'Figure1f_percentage.pdf')
        plt.savefig(outf2)
        plt.close()

        T.open_path_and_file(outdir)

        pass

    def Figure2a(self):
        outdir = join(self.this_class_png, 'Figure2a')
        T.mk_dir(outdir)
        fpath = join(self.this_class_tif,'Figure1c/NDVI_anomaly_drought_season.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
        df = T.spatial_dics_to_df({'delta':spatial_dict})
        df = Dataframe_func(df).df
        T.print_head_n(df)
        gez_list = T.get_df_unique_val_list(df, 'GEZ')
        gez_list = list(gez_list)
        gez_list.remove('Water')
        print(gez_list)
        count_list = []
        count_list_humid = []
        count_list_arid = []
        delta_list = []
        # delta_max_list = []
        delta_25_list = []
        bar_delta_list = []
        for gez in gez_list:
            df_gez = df[df['GEZ'] == gez]
            df_gez_humid = df_gez[df_gez['AI_class'] == 'Humid']
            df_gez_arid = df_gez[df_gez['AI_class'] == 'Arid']
            # T.print_head_n(df_gez);exit()
            delta = df_gez['delta'].tolist()
            count_list.append(len(df_gez))
            count_list_humid.append(len(df_gez_humid))
            count_list_arid.append(len(df_gez_arid))
            delta_mean = np.nanmean(delta)
            delta_25_percentile = np.percentile(delta,25)
            delta_75_percentile = np.percentile(delta,75)
            # delta_max = np.nanmax(delta)
            # delta_min = np.nanmin(delta)
            # count = len(df_gez)
            # count_list.append(count)
            delta_list.append(delta_mean)
            bar_delta = delta_75_percentile - delta_25_percentile
            bar_delta_list.append(bar_delta)
            delta_25_list.append(delta_25_percentile)
        # plt.scatter(gez_list,delta_max_list,label='max')
        # plt.scatter(gez_list,delta_min_list,label='min')
        plt.bar(gez_list,bar_delta_list,bottom=delta_25_list)
        plt.scatter(gez_list,delta_list,label='mean')
        # plt.legend()
        plt.xticks(rotation=90)
        plt.tight_layout()

        outf = join(outdir, 'delta_bar.pdf')
        plt.savefig(outf)
        # plt.show()
        plt.close()

        plt.figure()
        # plt.bar(gez_list,count_list)
        # print(count_list_humid);exit()
        plt.bar(gez_list,count_list_humid,color='b')
        plt.bar(gez_list,count_list_arid,bottom=count_list_humid,color='r')
        plt.xticks(rotation=90)
        plt.tight_layout()
        outf = join(outdir,'count.pdf')
        plt.savefig(outf)
        plt.close()
        # plt.show()
        T.open_path_and_file(outdir)
        pass

    def Figure2b(self):
        outdir = join(self.this_class_png, 'Figure2b')
        T.mk_dir(outdir)
        tif = join(self.this_class_tif, 'Figure1c/NDVI_anomaly_drought_season.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(tif)

        MAT_tif = join(data_root, r'CRU_tmp\mat', 'mat_gs.tif')
        MAT_dict = DIC_and_TIF().spatial_tif_to_dic(MAT_tif)

        MAP_tif = join(data_root, r'CRU_precip\map', 'map.tif')
        MAP_dict = DIC_and_TIF().spatial_tif_to_dic(MAP_tif)
        spatial_dict_all = {
            'compensation_excerbation': spatial_dict,
            'MAT': MAT_dict,
            'MAP': MAP_dict
        }
        df = T.spatial_dics_to_df(spatial_dict_all)
        df = df.dropna(how='any')
        map = df['MAP'].tolist()
        # plt.hist(map, bins=100, range=(0, 1500), zorder=-99, color='gray', alpha=0.5)
        # plt.show()
        T.print_head_n(df)
        # exit()
        # df = df.dropna(how='any')
        bins_Topt_MAT_delta = np.arange(-5, 36, 1)
        bins_MAP = np.arange(0, 4001, 100)
        plt.figure(figsize=(7, 6))
        col_name = 'compensation_excerbation'
        df_group_Topt_MAT_delta, bins_list_str_Topt_MAT_delta = T.df_bin(df, 'MAT', bins_Topt_MAT_delta)
        for name_Topt_MAT_delta, df_group_i_Topt_MAT_delta in df_group_Topt_MAT_delta:
            y_pos = name_Topt_MAT_delta[0].left
            df_group_MAP, bins_list_str_MAP = T.df_bin(df_group_i_Topt_MAT_delta, 'MAP', bins_MAP)
            for name_MAP, df_group_i_MAP in df_group_MAP:
                x_pos = name_MAP[0].left
                vals = df_group_i_MAP[col_name].tolist()
                # if len(vals) < 10:
                #     continue
                if T.is_all_nan(vals):
                    continue
                mean = np.nanmean(vals)
                ax=plt.scatter(y_pos, x_pos, s=40, c=mean, vmin=-0.7, vmax=0.7, cmap='RdBu', marker='s', linewidths=0)
                # plt.scatter(x_pos,y_pos,s=13,c=mean,vmin=-0.01,vmax=0.01,cmap='RdBu',marker='s',linewidths=0)
        plt.colorbar()
        plt.xlabel('MAT')
        plt.ylabel('MAP')
        plt.ylim(-100, 4500)
        plt.xlim(-13, 35)
        # outf = join(outdir, 'compensation_excerbation_MAT_MAP.pdf')
        outf = join(outdir, 'colorbar.pdf')
        plt.savefig(outf)
        plt.close()
        # plt.show()
        # exit()
        pass

    def Figure3a(self,df):
        # exit()
        outdir = join(self.this_class_png, 'Figure3a')
        T.mkdir(outdir)
        # df = df[df['aridity_index'] <= 3]
        # df = df[df['aridity_index_new'] <= 3]
        # df = df[df['Temperature-anomaly_detrend_drought_growing_season'] <= 2]
        # df = df[df['Temperature-anomaly_detrend_drought_growing_season'] >= -2]
        df = df.dropna(subset=['Temperature-anomaly_detrend'], how='any')
        T_anomaly_vals = df['Temperature-anomaly_detrend'].tolist()

        bin_range = np.linspace(0, 1, 41)
        T_quantile_bins = []
        AI_bins = np.linspace(0, 2.5, 26)
        # AI_bins = np.linspace(0, 1.6, 32)
        for b_i in bin_range:
            T_quantile_bins.append(np.quantile(T_anomaly_vals, b_i))
            # AI_quantile_bins.append(np.quantile(AI_vals,b_i))
        # print(T_quantile_bins);exit()

        df_group_AI, _ = T.df_bin(df, 'aridity_index', AI_bins)
        # df_group_AI, _ = T.df_bin(df, 'aridity_index_new', AI_bins)

        matrix = []
        y_label_list = []
        plt.figure(figsize=(7, 3.5))
        for name_AI, df_group_AI_i in df_group_AI:
            matrix_i = []
            y_label = (name_AI[0].left + name_AI[0].right) / 2
            y_label = np.round(y_label, 2)
            y_label_list.append(y_label)
            x_label_list = []

            # df_group_T, _ = T.df_bin(df_group_AI_i, 'Temperature-anomaly_detrend_drought_growing_season', T_quantile_bins)
            df_group_T, _ = T.df_bin(df_group_AI_i, 'Temperature-anomaly_detrend', T_quantile_bins)
            # if len(df_group_T) != len(T_quantile_bins)-1:
            #     continue
            # print(len(T_quantile_bins))
            flag = 0

            for name_T, df_group_T_i in df_group_T:
                rt = df_group_T_i['NDVI-anomaly_detrend_drought_growing_season'].tolist()
                # rt = df_group_T_i['NDVI-anomaly_detrend'].tolist()
                rt_mean = np.nanmean(rt)
                matrix_i.append(rt_mean)
                # print(rt_mean)
                x_label = (name_T[0].left + name_T[0].right) / 2
                x_label = np.round(x_label, 2)
                x_label_list.append(x_label)
                plt.scatter(bin_range[flag], y_label, c=rt_mean, vmin=-.35, vmax=.35, cmap='RdBu', marker='s')
                # print(flag,rt_mean)
                flag += 1
        plt.ylabel('AI')
        plt.xlabel('T_anomaly_quantile')
        plt.colorbar()
        # plt.xticks(range(len(x_label_list)),x_label_list,rotation=90)
        # plt.yticks(range(len(y_label_list))[::-1],y_label_list)
        # plt.tight_layout()
        # plt.show()
        outf = join(outdir,'matrix.pdf')
        plt.savefig(outf)
        plt.close()
        T.open_path_and_file(outdir)
        pass

    def Figure3a_count(self,df):
        # exit()
        outdir = join(self.this_class_png, 'Figure3a')
        T.mkdir(outdir)
        # df = df[df['aridity_index'] <= 3]
        df = df[df['aridity_index_new'] <= 3]
        df = df[df['Temperature-anomaly_detrend_drought_growing_season'] <= 2]
        df = df[df['Temperature-anomaly_detrend_drought_growing_season'] >= -2]
        df = df.dropna(subset=['Temperature-anomaly_detrend_drought_growing_season'], how='any')
        T_anomaly_vals = df['Temperature-anomaly_detrend_drought_growing_season'].tolist()
        area_dict = DIC_and_TIF().calculate_pixel_area()
        df = T.add_spatial_dic_to_df(df, area_dict, 'area')

        bin_range = np.linspace(0, 1, 41)
        T_quantile_bins = []
        AI_bins = np.linspace(0, 2.5, 26)
        for b_i in bin_range:
            T_quantile_bins.append(np.quantile(T_anomaly_vals, b_i))
            # AI_quantile_bins.append(np.quantile(AI_vals,b_i))
        # print(T_quantile_bins);exit()

        df_group_AI, _ = T.df_bin(df, 'aridity_index_new', AI_bins)
        # df_group_AI, _ = T.df_bin(df, 'aridity_index', AI_bins)

        matrix = []
        all_vals = []
        y_label_list = []
        plt.figure(figsize=(7, 3.5))
        for name_AI, df_group_AI_i in df_group_AI:
            matrix_i = []
            y_label = (name_AI[0].left + name_AI[0].right) / 2
            y_label = np.round(y_label, 2)
            y_label_list.append(y_label)
            x_label_list = []

            df_group_T, _ = T.df_bin(df_group_AI_i, 'Temperature-anomaly_detrend_drought_growing_season', T_quantile_bins)
            # if len(df_group_T) != len(T_quantile_bins)-1:
            #     continue
            # print(len(T_quantile_bins))
            flag = 0

            for name_T, df_group_T_i in df_group_T:
                # rt = df_group_T_i['NDVI-anomaly_detrend_drought_growing_season'].tolist()
                rt = df_group_T_i['area'].tolist()
                rt = np.array(rt)
                rt = rt[~np.isnan(rt)]
                rt_mean = np.nansum(rt)
                rt_mean = np.log(rt_mean)
                # print(rt_mean)
                # rt_mean = np.nanmean(rt)
                matrix_i.append(rt_mean)
                all_vals.append(rt_mean)
                # print(rt_mean)
                x_label = (name_T[0].left + name_T[0].right) / 2
                x_label = np.round(x_label, 2)
                x_label_list.append(x_label)
                # plt.scatter(bin_range[flag], y_label, c=rt_mean, cmap='Blues', marker='s',vmin=0,vmax=400)
                plt.scatter(bin_range[flag], y_label, c=rt_mean, cmap='Blues', marker='s',vmin=22,vmax=28)
                # print(flag,rt_mean)
                flag += 1
        # plt.hist(all_vals,bins=100)
        # plt.show()
        plt.ylabel('AI')
        plt.xlabel('T_anomaly_quantile')
        plt.colorbar()
        # plt.xticks(range(len(x_label_list)),x_label_list,rotation=90)
        # plt.yticks(range(len(y_label_list))[::-1],y_label_list)
        # plt.tight_layout()
        # plt.show()
        # outf = join(outdir,'matrix_count_new.pdf')
        outf = join(outdir,'matrix_count_area.pdf')
        plt.savefig(outf)
        plt.close()
        T.open_path_and_file(outdir)
        pass

    def Figure3a_cdf(self):
        # exit()
        outdir = join(self.this_class_png, 'Figure3a')
        T.mkdir(outdir)
        # fpath = join(data_root,'Aridity_Index/aridity_index.tif')
        fpath = join(data_root,'Aridity_index_new/ai_05_rescale.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
        df = T.spatial_dics_to_df({'aridity_index1': spatial_dict})
        df = Dataframe_func(df).df
        area_dict = DIC_and_TIF().calculate_pixel_area()
        df = T.add_spatial_dic_to_df(df,area_dict,'area')
        # T.print_head_n(df);exit()
        # aridity_index1 = df['aridity_index1'].tolist()
        bins = np.linspace(0,2.5,100)
        df_group, bins_list_str = T.df_bin(df, 'aridity_index1', bins)
        sum_area_list = []
        sum_area_all = 0
        for name,df_group_i in df_group:
            vals = df_group_i['area'].tolist()
            sum_area = np.nansum(vals)
            sum_area_all += sum_area
            sum_area_list.append(sum_area_all)
        plt.plot(bins[1:],sum_area_list)
        plt.xlabel('AI')
        plt.ylabel('area')
        plt.tight_layout()
        # plt.show()
        outf = join(outdir,'matrix_cdf_area.pdf')
        # outf = join(outdir,'matrix_cdf.pdf')
        plt.savefig(outf)
        plt.close()
        T.open_path_and_file(outdir)
        pass


    def Figure3b(self):
        outdir = join(self.this_class_png, 'delta_area_ratio_bar')
        T.mk_dir(outdir)
        tif = join(self.this_class_tif, 'Figure1c/NDVI_anomaly_drought_season.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(tif)
        df = T.spatial_dics_to_df({'delta': spatial_dict})
        df = Dataframe_func(df).df
        threshold_list = [-np.inf, -0.5, -0.25, 0, 0.25, 0.5, np.inf]
        color_list = T.gen_colors(len(threshold_list) - 1)
        # threshold_list = [-np.inf,-0.2,-0.1,-0.05]
        # threshold_list = np.array(threshold_list)
        # threshold_list = threshold_list + 1
        plt.figure(figsize=(15 * centimeter_factor, 6 * centimeter_factor))
        bottom = 0
        y_list_all = []
        for i in range(len(threshold_list)):
            if i + 1 == len(threshold_list):
                break
            x_list = []
            y_list = []
            AI_bins = np.arange(0.1, 2.6, 0.1)
            df_group, bins_list_str = T.df_bin(df, 'aridity_index', AI_bins)
            for name, df_group_i in df_group:
                x = name[0].left
                df_group_i = df_group_i.dropna(subset=['delta'], how='any')
                vals = df_group_i['delta']
                vals = np.array(vals)
                threshold_left = threshold_list[i]
                threshold_right = threshold_list[i + 1]

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
            plt.bar(x_list, y_list, bottom=bottom, label=f'{threshold_list[i]}~{threshold_list[i + 1]}',
                    color=color_list[i], width=0.08)
            # width=0.1,edgecolor='k')
            bottom = bottom + y_list
            print(bottom)

        # plt.legend()
        plt.xticks(rotation=0)
        plt.ylim(0, 100)
        # plt.xlim(-1,2)
        plt.twinx()
        self.__AI_gradient_Drought_year_spatial_tif(tif)
        plt.tight_layout()

        outf = join(outdir,'delta_AI_gradient.pdf')
        plt.savefig(outf)
        plt.close()
        T.open_path_and_file(outdir)
        # plt.show()

    def Figure4_values_boxplot(self,df):
        outdir = join(self.this_class_png, 'Figure4_values')
        T.mk_dir(outdir)
        df = df.dropna(subset=['early_range'], how='any')
        df = df.drop('drought_season', axis=1)
        # T.print_head_n(df)
        # exit()
        drought_season_list = []
        for i,row in df.iterrows():
            drought_mon = row['drought_mon']
            growing_season = row['growing_season']
            early_range = row['early_range']
            mid_range = row['mid_range']
            late_range = row['late_range']
            if not drought_mon in growing_season:
                drought_season_list.append(np.nan)
                continue
            if drought_mon in early_range:
                drought_season = 'early'
            elif drought_mon in mid_range:
                drought_season = 'mid'
            elif drought_mon in late_range:
                drought_season = 'late'
            else:
                raise
            drought_season_list.append(drought_season)
        df['drought_season'] = drought_season_list
        T.print_head_n(df)
        label_list = []
        mean_values_list = []
        mean_pos_values_list = []
        mean_neg_values_list = []
        err_list = []
        flag_list = []

        flag = 0
        plt.figure(figsize=(15 * centimeter_factor, 10 * centimeter_factor))

        NDVI_vals_list = []
        label_list = []
        for AI_class in global_AI_class_list:
            df_AI = df[df['AI_class'] == AI_class]
            for season in ['early', 'mid', 'late']:
                df_season = df_AI[df_AI['drought_season'] == season]
                for drt in global_drought_type_list:
                    df_drt = df_season[df_season['drought_type'] == drt]
                    NDVI_vals = df_drt['NDVI-anomaly_detrend_drought_growing_season'].tolist()
                    NDVI_vals = np.array(NDVI_vals)
                    label = f'{drt}_{season}_{AI_class}'
                    label_list.append(label)
                    NDVI_vals_list.append(NDVI_vals)

                    # plt.plot([mean_neg,mean,mean_pos],[flag] * 3,color='k')
                    # flag = flag + 1

        plt.boxplot(NDVI_vals_list,labels=label_list,patch_artist=True,showfliers=False)
        plt.xticks(rotation=90)
        plt.tight_layout()
        # plt.show()
        outf = join(outdir, 'Figure4_values_boxplot.pdf')
        plt.savefig(outf)
        plt.close()
        T.open_path_and_file(outdir)

        pass

    def Figure4_values(self,df):
        outdir = join(self.this_class_png, 'Figure4_values')
        T.mk_dir(outdir)
        df = df.dropna(subset=['early_range'], how='any')
        df = df.drop('drought_season', axis=1)
        # T.print_head_n(df)
        # exit()
        drought_season_list = []
        for i,row in df.iterrows():
            drought_mon = row['drought_mon']
            growing_season = row['growing_season']
            early_range = row['early_range']
            mid_range = row['mid_range']
            late_range = row['late_range']
            if not drought_mon in growing_season:
                drought_season_list.append(np.nan)
                continue
            if drought_mon in early_range:
                drought_season = 'early'
            elif drought_mon in mid_range:
                drought_season = 'mid'
            elif drought_mon in late_range:
                drought_season = 'late'
            else:
                raise
            drought_season_list.append(drought_season)
        df['drought_season'] = drought_season_list
        T.print_head_n(df)
        label_list = []
        mean_values_list = []
        mean_pos_values_list = []
        mean_neg_values_list = []
        err_list = []
        flag_list = []

        flag = 0
        plt.figure(figsize=(15 * centimeter_factor, 6 * centimeter_factor))

        for AI_class in global_AI_class_list:
            df_AI = df[df['AI_class'] == AI_class]
            for season in ['early', 'mid', 'late']:
                df_season = df_AI[df_AI['drought_season'] == season]
                for drt in global_drought_type_list:
                    df_drt = df_season[df_season['drought_type'] == drt]
                    NDVI_vals = df_drt['NDVI-anomaly_detrend_drought_growing_season'].tolist()
                    NDVI_vals = np.array(NDVI_vals)
                    NDVI_vals_pos = NDVI_vals[NDVI_vals > 0]
                    NDVI_vals_neg = NDVI_vals[NDVI_vals < 0]
                    label = f'{drt}_{season}_{AI_class}'
                    mean = np.nanmean(NDVI_vals)
                    mean_pos = np.nanmean(NDVI_vals_pos)
                    mean_neg = np.nanmean(NDVI_vals_neg)

                    # std = np.nanstd(NDVI_vals)
                    _,_,_,err = self.mean_confidence_interval(NDVI_vals)
                    err_list.append(err)
                    label_list.append(label)
                    mean_values_list.append(mean)
                    mean_pos_values_list.append(mean_pos)
                    mean_neg_values_list.append(mean_neg)
                    flag_list.append(flag)
                    plt.plot([mean_neg,mean,mean_pos],[flag] * 3,color='k')
                    flag = flag + 1

        # plt.barh(flag_list, mean_values_list,xerr=err_list, height=0.4)
        plt.scatter(mean_values_list, flag_list, color='k', s=40)
        plt.scatter(mean_pos_values_list, flag_list, color='r', s=20)
        plt.scatter(mean_neg_values_list, flag_list, color='b', s=20)
        plt.xlim(-0.8,0.4)
        # plt.xticks(rotation=45)
        plt.yticks(range(len(flag_list)), label_list)
        plt.tight_layout()
        # plt.show()
        outf = join(outdir, 'Figure4_values.pdf')
        plt.savefig(outf)
        plt.close()
        T.open_path_and_file(outdir)

        pass

    def Figure4_values_std(self,df):
        outdir = join(self.this_class_png, 'Figure4_values')
        # T.print_head_n(df);exit()
        T.mk_dir(outdir)
        df = df.dropna(subset=['early_range'], how='any')
        df = df.drop('drought_season', axis=1)
        # T.print_head_n(df)
        # exit()
        drought_season_list = []
        for i,row in df.iterrows():
            drought_mon = row['drought_mon']
            growing_season = row['growing_season']
            early_range = row['early_range']
            mid_range = row['mid_range']
            late_range = row['late_range']
            if not drought_mon in growing_season:
                drought_season_list.append(np.nan)
                continue
            if drought_mon in early_range:
                drought_season = 'early'
            elif drought_mon in mid_range:
                drought_season = 'mid'
            elif drought_mon in late_range:
                drought_season = 'late'
            else:
                raise
            drought_season_list.append(drought_season)
        df['drought_season'] = drought_season_list
        T.print_head_n(df)
        label_list = []
        mean_values_list = []
        mean_pos_values_list = []
        mean_neg_values_list = []
        err_list = []
        flag_list = []

        flag = 0
        plt.figure(figsize=(15 * centimeter_factor, 6 * centimeter_factor))

        for AI_class in global_AI_class_list:
            df_AI = df[df['AI_class'] == AI_class]
            for season in ['early', 'mid', 'late']:
                df_season = df_AI[df_AI['drought_season'] == season]
                for drt in global_drought_type_list:
                    df_drt = df_season[df_season['drought_type'] == drt]
                    NDVI_vals = df_drt['NDVI-anomaly_detrend_drought_growing_season'].tolist()
                    NDVI_vals = np.array(NDVI_vals)
                    NDVI_vals_std = np.nanstd(NDVI_vals)
                    label = f'{drt}_{season}_{AI_class}'
                    mean = np.nanmean(NDVI_vals)
                    err = NDVI_vals_std
                    # _,_,_,err = self.mean_confidence_interval(NDVI_vals)
                    err_list.append(err)
                    label_list.append(label)
                    mean_values_list.append(mean)
                    mean_neg = mean - err
                    mean_pos = mean + err
                    mean_pos_values_list.append(mean_pos)
                    mean_neg_values_list.append(mean_neg)
                    flag_list.append(flag)
                    plt.plot([mean_neg,mean,mean_pos],[flag] * 3,color='k')
                    flag = flag + 1

        # plt.barh(flag_list, mean_values_list,xerr=err_list, height=0.4)
        plt.scatter(mean_values_list, flag_list, color='k', s=40)
        plt.scatter(mean_pos_values_list, flag_list, color='r', s=20)
        plt.scatter(mean_neg_values_list, flag_list, color='b', s=20)
        plt.xlim(-2,1)
        # plt.xticks(rotation=45)
        plt.yticks(range(len(flag_list)), label_list)
        plt.tight_layout()
        # plt.show()
        outf = join(outdir, 'Figure4_values_std.pdf')
        plt.savefig(outf)
        plt.close()
        T.open_path_and_file(outdir)

        pass

    def Figure4_ratio(self,df):
        outdir = join(self.this_class_png, 'Figure4_ratio')
        T.mk_dir(outdir)
        df = df.dropna(subset=['early_range'], how='any')
        df = df.drop('drought_season', axis=1)
        # T.print_head_n(df)
        # exit()
        area_dict = DIC_and_TIF().calculate_pixel_area()
        df = T.add_spatial_dic_to_df(df, area_dict, 'area')
        drought_season_list = []
        for i,row in df.iterrows():
            drought_mon = row['drought_mon']
            growing_season = row['growing_season']
            early_range = row['early_range']
            mid_range = row['mid_range']
            late_range = row['late_range']
            if not drought_mon in growing_season:
                drought_season_list.append(np.nan)
                continue
            if drought_mon in early_range:
                drought_season = 'early'
            elif drought_mon in mid_range:
                drought_season = 'mid'
            elif drought_mon in late_range:
                drought_season = 'late'
            else:
                raise
            drought_season_list.append(drought_season)
        df['drought_season'] = drought_season_list
        T.print_head_n(df)
        for AI_class in global_AI_class_list:
            df_AI = df[df['AI_class'] == AI_class]
            for season in ['early', 'mid', 'late']:
                df_season = df_AI[df_AI['drought_season'] == season]
                for drt in global_drought_type_list:
                    df_drt = df_season[df_season['drought_type'] == drt]
                    df_drt = df_drt.dropna(subset=['NDVI-anomaly_detrend_drought_growing_season'], how='any')
                    NDVI_vals = df_drt['NDVI-anomaly_detrend_drought_growing_season'].tolist()
                    area_vals = df_drt['area'].tolist()
                    NDVI_vals = np.array(NDVI_vals)
                    area_vals = np.array(area_vals)
                    # print(len(NDVI_vals))
                    # NDVI_vals = NDVI_vals[~np.isnan(NDVI_vals)]
                    # total_area = np.nansum(area_vals)
                    # pos_area1 = area_vals[NDVI_vals > 0]
                    # neg_area1 = area_vals[NDVI_vals < 0]
                    # pos_area2 = area_vals[NDVI_vals > .5]
                    # neg_area2 = area_vals[NDVI_vals < -.5]
                    # positive_ratio1 = np.nansum(pos_area1) / total_area
                    # neagative_ratio1 = np.nansum(neg_area1) / total_area
                    # positive_ratio2 = np.nansum(pos_area2) / total_area
                    # neagative_ratio2 = np.nansum(neg_area2) / total_area

                    positive_ratio1 = len(NDVI_vals[NDVI_vals > 0]) / len(NDVI_vals)
                    neagative_ratio1 = len(NDVI_vals[NDVI_vals < -0]) / len(NDVI_vals)
                    positive_ratio2 = len(NDVI_vals[NDVI_vals > .5]) / len(NDVI_vals)
                    neagative_ratio2 = len(NDVI_vals[NDVI_vals < -.5]) / len(NDVI_vals)
                    label = f'{drt}_{season}_{AI_class}'
                    neagative_ratio1 = -neagative_ratio1
                    neagative_ratio2 = -neagative_ratio2
                    plt.bar(label,positive_ratio2,color='b')
                    plt.bar(label,neagative_ratio2,color='r')
                    plt.bar(label,positive_ratio1,color='none',lw=1,edgecolor='k')
                    plt.bar(label,neagative_ratio1,color='none',lw=1,edgecolor='k')
        plt.xticks(rotation=45,ha='right')
        plt.ylim(-0.8,0.6)
        plt.tight_layout()
        plt.show()
        # outf = join(outdir, 'Figure4_ratio.pdf')
        # plt.savefig(outf)
        # plt.close()
        # T.open_path_and_file(outdir)

        pass

    def Figure4_values_ratio(self,df):

        pass

    def Figure4_trajectory(self,df):
        df = df.dropna(subset=['drought_season'], how='any')
        T.print_head_n(df)
        drought_season_list = ['early', 'mid', 'late']
        # print(drought_season_list)
        for AI_class in global_AI_class_list:
            df_AI = df[df['AI_class'] == AI_class]
            for season in drought_season_list:
                df_season = df_AI[df_AI['drought_season'] == season]
                plt.figure(figsize=(15 * centimeter_factor, 6 * centimeter_factor))
                for drt in global_drought_type_list:
                    df_drt = df_season[df_season['drought_type'] == drt]
                    dict_i = {i:[] for i in range(1,13)}
                    for i,row in tqdm(df_drt.iterrows(),total=len(df_drt)):
                        NDVI_vals = row['NDVI-anomaly_detrend_drought_year_growing_season']
                        growing_season = row['growing_season']
                        dict_j = T.dict_zip(growing_season, NDVI_vals)
                        for k in dict_j:
                            dict_i[k].append(dict_j[k])

                    NDVI_vals_mean_list = []
                    err_list = []
                    month_list = list(range(1,13))
                    for i in month_list:
                        NDVI_vals = dict_i[i]
                        NDVI_vals = np.array(NDVI_vals)
                        NDVI_vals = NDVI_vals[~np.isnan(NDVI_vals)]
                        NDVI_vals_mean = np.nanmean(NDVI_vals)
                        _,_,_,err = self.mean_confidence_interval(NDVI_vals)
                        # err = np.nanstd(NDVI_vals) / 8.
                        NDVI_vals_mean_list.append(NDVI_vals_mean)
                        err_list.append(err)
                    err_list = np.array(err_list)
                    plt.plot(month_list,NDVI_vals_mean_list,label=f'{drt}_{season}_{AI_class}')
                    # plt.errorbar(np.arange(5,13),NDVI_vals_mean_list, yerr=err_list)
                    plt.fill_between(month_list, NDVI_vals_mean_list - err_list, NDVI_vals_mean_list + err_list, alpha=0.2)
                    plt.xlabel('Month')
                    plt.ylim(-0.85,.25)
                plt.legend()
        plt.show()

        pass

    def Figure5(self,df):


        pass

    def Figure_S6(self,df):
        AI_tif = join(data_root,'Aridity_Index/aridity_index.tif')
        outdir = join(self.this_class_tif,'Figure_S6')
        outdir_png = join(self.this_class_png,'Figure_S6')
        T.mkdir(outdir)
        T.mkdir(outdir_png)
        AI_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(AI_tif)
        delta_tif = join(self.this_class_tif,'Figure1c','NDVI_anomaly_drought_season.tif')
        delta_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(delta_tif)
        all_dicts = {'AI_':AI_spatial_dict,'delta':delta_spatial_dict}
        df = T.spatial_dics_to_df(all_dicts)
        # df = df.dropna(subset=['delta'], how='any')
        df = Dataframe_func(df).df
        AI_bins = np.array([0,0.05,0.2,0.5,0.65,0.8,1.5,np.inf])
        df_group, bins_list_str = T.df_bin(df, 'AI_', AI_bins)
        spatial_dict = {}
        for name,df_group_i in df_group:
            vals = df_group_i['delta'].tolist()
            mean = np.nanmean(vals)
            pix_list = df_group_i['pix'].tolist()
            for pix in pix_list:
                spatial_dict[pix] = mean
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        outtif = join(outdir,'Figure_S6.tif')
        DIC_and_TIF().arr_to_tif(arr,outtif)
        m, ret = Plot().plot_ortho(outtif, vmin=-.3, vmax=.3, cmap='RdBu')
        # m,ret = Plot().plot_ortho_significance_scatter(fpath,vmin=-1,vmax=1,cmap='RdBu')
        outpng = join(outdir_png,'Figure_S6.png')
        plt.savefig(outpng, dpi=900)
        T.open_path_and_file(outdir_png)

        pass

    def Figure_S6_1(self,df):
        AI_tif = join(data_root,'Aridity_Index/aridity_index.tif')
        outdir = join(self.this_class_tif,'Figure_S6')
        outdir_png = join(self.this_class_png,'Figure_S6')
        T.mkdir(outdir)
        T.mkdir(outdir_png)
        AI_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(AI_tif)
        delta_tif = join(self.this_class_tif,'Figure1c','NDVI_anomaly_drought_season.tif')
        delta_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(delta_tif)
        all_dicts = {'AI_':AI_spatial_dict,'delta':delta_spatial_dict}
        df = T.spatial_dics_to_df(all_dicts)
        # df = df.dropna(subset=['delta'], how='any')
        df = Dataframe_func(df).df
        AI_bins = np.array([0,0.05,0.2,0.5,0.65,0.8,1.5,np.inf])
        df_group, bins_list_str = T.df_bin(df, 'AI_', AI_bins)
        # print(len(df_group));exit()
        spatial_dict = {}
        flag = 1
        for name,df_group_i in df_group:
            vals = df_group_i['AI_'].tolist()
            # mean = flag
            mean = np.nanmax(vals)
            flag += 1
            pix_list = df_group_i['pix'].tolist()
            for pix in pix_list:
                spatial_dict[pix] = mean
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        outtif = join(outdir,'Figure_S6_1.tif')
        DIC_and_TIF().arr_to_tif(arr,outtif)
        m, ret = Plot().plot_ortho(outtif, vmin=0, vmax=2.5, cmap='RdBu')
        # m,ret = Plot().plot_ortho_significance_scatter(fpath,vmin=-1,vmax=1,cmap='RdBu')
        outpng = join(outdir_png,'Figure_S6_1.png')
        plt.savefig(outpng, dpi=900)
        T.open_path_and_file(outdir_png)

        pass

    def SOS_drought_timing_AI(self,df):
        df = df.dropna(subset=['drought_season'], how='any')
        outdir = join(self.this_class_png,'SOS_drought_timing_AI')
        T.mkdir(outdir)
        # drought_type = 'normal-drought'
        # drought_type = 'hot-drought'
        drought_type = 'All'
        # df = df[df['drought_type']==drought_type]
        print('------------')
        T.print_head_n(df)
        bin_y = np.linspace(0, 2.6, 14)
        # print(bin_y);exit()
        bin_x = np.linspace(-20, 20, 21)
        val_col_name = 'NDVI-anomaly_detrend_drought_growing_season'
        col_name_x = 'SOS'
        col_name_y = 'aridity_index'
        for season in ['early','mid','late']:
            print(season)
            df_season = df[df['drought_season'] == season]
            # df_group, bins_list_str = T.df_bin(df_season, 'SOS', sos_bins)
            matrix_dict,x_ticks_list,y_ticks_list = T.df_bin_2d(df_season,val_col_name,col_name_x,col_name_y,
                                                                bin_x,bin_y,round_x=10,round_y=10)
            # print(y_ticks_list);exit()
            # print(x_ticks_list)
            # print(y_ticks_list)
            plt.figure(figsize=(3.0776,1.6736))
            # T.plot_df_bin_2d_matrix(matrix_dict,-0.6,0.6,x_ticks_list,y_ticks_list)
            # T.plot_df_bin_2d_scatter(matrix_dict,-0.6,0.6,x_ticks_list,y_ticks_list,s=200)
            T.plot_df_bin_2d_scatter(matrix_dict,-0.6,0.6,x_ticks_list,y_ticks_list,s=36.75269376)
            plt.colorbar()
            plt.title(f'{season}_{drought_type}')
            outf = join(outdir,f'{season}_{drought_type}.pdf')
            # plt.show()
            # pause()
            plt.xlim(-20,20)
            plt.ylim(0,2.6)
            plt.savefig(outf)
            plt.close()
            # exit()

    def __AI_gradient_Drought_year_spatial_tif(self,delta_fpath):
        spatial_dics = {}
        spatial_dics['delta'] = DIC_and_TIF().spatial_tif_to_dic(delta_fpath)
        df = T.spatial_dics_to_df(spatial_dics)
        df = Dataframe_func(df).df
        print('-----')
        # ELI_class_list = global_ELI_class_list
        key = 'delta'
        # plt.figure(figsize=(9*centimeter_factor,6*centimeter_factor))
        AI_bins = np.arange(0.1, 2.6, 0.1)
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


    def add_NDVI_anomaly_to_df(self,df):
        NDVI_dict, var_name, _ = Load_Data().NDVI_anomaly_detrend()
        NDVI_drought_growing_season_mean_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            growing_season = row['growing_season']
            if type(growing_season) == float:
                NDVI_drought_growing_season_mean_list.append(np.nan)
                continue
            drought_year = row['drought_year']
            NDVI_vals = NDVI_dict[pix]
            NDVI_vals_reshape = np.array(NDVI_vals).reshape(-1, 12)
            NDVI_vals_annual_dict = T.dict_zip(global_year_range_list, NDVI_vals_reshape)
            NDVI_drought_year = NDVI_vals_annual_dict[drought_year]
            NDVI_drought_growing_season = NDVI_drought_year[growing_season - 1]
            NDVI_drought_growing_season_mean = np.nanmean(NDVI_drought_growing_season)
            NDVI_drought_growing_season_mean_list.append(NDVI_drought_growing_season_mean)
        df[f'{var_name}_drought_growing_season'] = NDVI_drought_growing_season_mean_list

        return df

    def add_NDVI_anomaly_drought_year_to_df(self,df):
        NDVI_dict, var_name, _ = Load_Data().NDVI_anomaly_detrend()
        NDVI_drought_growing_season_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            growing_season = row['growing_season']
            if type(growing_season) == float:
                NDVI_drought_growing_season_list.append(np.nan)
                continue
            drought_year = row['drought_year']
            NDVI_vals = NDVI_dict[pix]
            NDVI_vals_reshape = np.array(NDVI_vals).reshape(-1, 12)
            NDVI_vals_annual_dict = T.dict_zip(global_year_range_list, NDVI_vals_reshape)
            NDVI_drought_year = NDVI_vals_annual_dict[drought_year]
            NDVI_drought_growing_season = NDVI_drought_year[growing_season - 1]
            NDVI_drought_growing_season_list.append(NDVI_drought_growing_season)
        df[f'{var_name}_drought_year_growing_season'] = NDVI_drought_growing_season_list

        return df

    def add_NDVI_anomaly_whole_year_to_df(self,df):
        NDVI_dict, var_name, _ = Load_Data().NDVI_anomaly_detrend()
        NDVI_drought_growing_season_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            NDVI_vals = NDVI_dict[pix]
            NDVI_vals_reshape = np.array(NDVI_vals).reshape(-1, 12)
            NDVI_vals_annual_dict = T.dict_zip(global_year_range_list, NDVI_vals_reshape)
            NDVI_drought_year = NDVI_vals_annual_dict[drought_year]
            NDVI_drought_growing_season_list.append(NDVI_drought_year)
        df[f'{var_name}_drought_year_whole_year'] = NDVI_drought_growing_season_list

        return df

    def add_T_anomaly_to_df(self, df):
        NDVI_dict, var_name, _ = Load_Data().Temperature_anomaly_detrend()
        NDVI_drought_growing_season_mean_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            growing_season = row['growing_season']
            if type(growing_season) == float:
                NDVI_drought_growing_season_mean_list.append(np.nan)
                continue
            drought_year = row['drought_year']
            NDVI_vals = NDVI_dict[pix]
            NDVI_vals_reshape = np.array(NDVI_vals).reshape(-1, 12)
            NDVI_vals_annual_dict = T.dict_zip(global_year_range_list, NDVI_vals_reshape)
            NDVI_drought_year = NDVI_vals_annual_dict[drought_year]
            NDVI_drought_growing_season = NDVI_drought_year[growing_season - 1]
            NDVI_drought_growing_season_mean = np.nanmean(NDVI_drought_growing_season)
            NDVI_drought_growing_season_mean_list.append(NDVI_drought_growing_season_mean)
        df[f'{var_name}_drought_growing_season'] = NDVI_drought_growing_season_mean_list
        return df

    def add_NDVI_percentage_to_df(self,df):
        NDVI_dict, var_name, _ = Load_Data().NDVI_percentage()
        NDVI_drought_growing_season_mean_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            growing_season = row['growing_season']
            if type(growing_season) == float:
                NDVI_drought_growing_season_mean_list.append(np.nan)
                continue
            drought_year = row['drought_year']
            NDVI_vals = NDVI_dict[pix]
            NDVI_vals_reshape = np.array(NDVI_vals).reshape(-1, 12)
            NDVI_vals_annual_dict = T.dict_zip(global_year_range_list, NDVI_vals_reshape)
            NDVI_drought_year = NDVI_vals_annual_dict[drought_year]
            NDVI_drought_growing_season = NDVI_drought_year[growing_season - 1]
            NDVI_drought_growing_season_mean = np.nanmean(NDVI_drought_growing_season)
            NDVI_drought_growing_season_mean_list.append(NDVI_drought_growing_season_mean)
        df[f'{var_name}_drought_growing_season'] = NDVI_drought_growing_season_mean_list

        return df

    def add_NDVI_percentage_drought_year_to_df(self,df):
        NDVI_dict, var_name, _ = Load_Data().NDVI_percentage()
        NDVI_drought_growing_season_mean_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            growing_season = row['growing_season']
            if type(growing_season) == float:
                NDVI_drought_growing_season_mean_list.append(np.nan)
                continue
            drought_year = row['drought_year']
            NDVI_vals = NDVI_dict[pix]
            NDVI_vals_reshape = np.array(NDVI_vals).reshape(-1, 12)
            NDVI_vals_annual_dict = T.dict_zip(global_year_range_list, NDVI_vals_reshape)
            NDVI_drought_year = NDVI_vals_annual_dict[drought_year]
            NDVI_drought_growing_season_mean_list.append(NDVI_drought_year)
        df[f'{var_name}_drought_year_vals'] = NDVI_drought_growing_season_mean_list

        return df

    def add_early_peak_late_period_to_df(self,df):
        import analysis
        fpath = join(analysis.Longterm_Phenology().this_class_arr,'early_peak_late_period','early_peak_late_period.npy')
        early_peak_late_period_dict = T.load_npy(fpath)
        early_range_list = []
        mid_range_list = []
        late_range_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            if not pix in early_peak_late_period_dict:
                early_range_list.append(np.nan)
                mid_range_list.append(np.nan)
                late_range_list.append(np.nan)
                continue
            early_peak_late_period_dict_i = early_peak_late_period_dict[pix]
            early_range = early_peak_late_period_dict_i['early_range']
            mid_range = early_peak_late_period_dict_i['mid_range']
            late_range = early_peak_late_period_dict_i['late_range']
            early_range_list.append(early_range)
            mid_range_list.append(mid_range)
            late_range_list.append(late_range)
        df['early_range'] = early_range_list
        df['mid_range'] = mid_range_list
        df['late_range'] = late_range_list

        return df

    def add_drought_season_to_df(self,df):
        df = df.drop('drought_season', axis=1)
        drought_season_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            drought_mon = row['drought_mon']
            growing_season = row['growing_season']
            early_range = row['early_range']
            mid_range = row['mid_range']
            late_range = row['late_range']
            if type(early_range) == float:
                drought_season_list.append(np.nan)
                continue
            if not drought_mon in growing_season:
                drought_season_list.append(np.nan)
                continue
            if drought_mon in early_range:
                drought_season = 'early'
            elif drought_mon in mid_range:
                drought_season = 'mid'
            elif drought_mon in late_range:
                drought_season = 'late'
            else:
                raise
            drought_season_list.append(drought_season)
        df['drought_season'] = drought_season_list
        return df

    def add_AI_new_to_df(self, df):
        f = join(data_root, 'Aridity_index_new/ai_05_rescale.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'aridity_index_new')
        return df

    def add_AI_new_class_to_df(self, df):
        AI_class_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            AI = row['aridity_index_new']
            if AI < 0.65:
                AI_class = 'Arid'
            elif AI >= 0.65:
                AI_class = 'Humid'
            else:
                AI_class = np.nan
            AI_class_list.append(AI_class)
        df['AI_class_new'] = AI_class_list
        return df

    def add_eos(self,df):
        import analysis
        pheno_str = 'late_end'
        fpath = join(analysis.Phenology().this_class_arr, 'phenology_df/phenology_df.df')
        phenology_df = T.load_df(fpath)
        cols = list(phenology_df.columns)
        print(cols)
        # exit()
        pheno_spatial_dict = {}
        for i, row in phenology_df.iterrows():
            pix = row['pix']
            early_start = row[pheno_str]
            early_start_dict = dict(early_start)
            phenology_anomaly_dict = self.phenology_anomaly_dict(early_start_dict)
            pheno_spatial_dict[pix] = phenology_anomaly_dict
        pheno_val_list = []
        for i, row in tqdm(df.iterrows(),total=len(df)):
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

    def phenology_anomaly_dict(self,phenology_dict):
        vals = list(phenology_dict.values())
        mean = np.nanmean(vals)
        phenology_anomaly_dict = {}
        for year in phenology_dict:
            val = phenology_dict[year]
            anomaly = val - mean
            phenology_anomaly_dict[year] = anomaly
        return phenology_anomaly_dict

    def mean_confidence_interval(self, data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return m, m - h, m + h,h

    def season_excerbation_alleviation_ratio_tif(self,df):
        df = df.dropna(subset=['drought_season'])
        drought_season_list = T.get_df_unique_val_list(df, 'drought_season')
        # print(drought_season_list)
        # T.print_head_n(df)

        # exit()
        outdir = join(self.this_class_tif, 'season_excerbation_alleviation_ratio')
        T.mk_dir(outdir)
        drought_season_list = ['early', 'mid', 'late']
        ELI_class_list = global_ELI_class_list[::-1]
        NDVI_col_name = 'NDVI-percentage_drought_year_vals'

        for season in drought_season_list:
            df_season = df[df['drought_season'] == season]
            # season_range = df_season[f'{season}_range'].tolist()
            # print(season_range);exit()
            df_pix_dict = T.df_groupby(df_season, 'pix')
            spatial_dict_normal = {}
            spatial_dict_hot = {}
            for pix in tqdm(df_pix_dict):
                df_pix = df_pix_dict[pix]
                df_hot = df_pix[df_pix['drought_type'] == 'hot-drought']
                df_normal = df_pix[df_pix['drought_type'] == 'normal-drought']
                df_hot_season_range = df_hot[f'{season}_range'].tolist()
                df_hot_season_range = np.array(df_hot_season_range) - 1
                df_normal_season_range = df_normal[f'{season}_range'].tolist()
                df_normal_season_range = np.array(df_normal_season_range) - 1
                # T.print_head_n(df_hot)
                # print(df_hot_season_range);exit()
                NDVI_vals_hot = df_hot[NDVI_col_name].tolist()
                NDVI_vals_hot = np.array(NDVI_vals_hot)
                NDVI_vals_hot_pix_mean = np.nanmean(NDVI_vals_hot, axis=0)

                # print(len(NDVI_vals_hot));exit()
                if len(df_hot_season_range) == 0:
                    NDVI_vals_hot_vals_mean = np.nan
                else:
                    NDVI_vals_hot_vals = NDVI_vals_hot_pix_mean[df_hot_season_range]
                    NDVI_vals_hot_vals_mean = np.nanmean(NDVI_vals_hot_vals)

                NDVI_vals_normal = df_normal[NDVI_col_name].tolist()
                NDVI_vals_normal = np.array(NDVI_vals_normal)
                NDVI_vals_normal_pix_mean = np.nanmean(NDVI_vals_normal,axis=0)
                # print(NDVI_vals_normal_pix_mean)
                if len(df_normal_season_range) == 0:
                    NDVI_vals_normal_vals_mean = np.nan
                else:
                    # print(df_normal_season_range)
                    # print(NDVI_vals_normal)
                    NDVI_vals_normal_vals = NDVI_vals_normal_pix_mean[df_normal_season_range]
                    NDVI_vals_normal_vals_mean = np.nanmean(NDVI_vals_normal_vals)

                drought_season_vals_hot_mean = NDVI_vals_hot_vals_mean
                drought_season_vals_normal_mean = NDVI_vals_normal_vals_mean

                spatial_dict_hot[pix] = drought_season_vals_hot_mean
                spatial_dict_normal[pix] = drought_season_vals_normal_mean
            arr_hot = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_hot)
            outf_hot = join(outdir, f'hot_{season}.tif')
            DIC_and_TIF().arr_to_tif(arr_hot, outf_hot)

            arr_normal = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_normal)
            outf_normal = join(outdir, f'normal_{season}.tif')
            DIC_and_TIF().arr_to_tif(arr_normal, outf_normal)

        pass

    def season_excerbation_alleviation_ratio_statistic(self):
        fdir = join(self.this_class_tif, 'season_excerbation_alleviation_ratio')
        outdir = join(self.this_class_arr, 'season_excerbation_alleviation_ratio_statistic')
        drt_list = ['hot', 'normal']
        T.mk_dir(outdir)
        spatial_dict_all = {}
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir, f)
            # arr = DIC_and_TIF().spatial_tif_to_arr(fpath)
            # plt.imshow(arr, cmap='RdBu', vmin=-0.2, vmax=0.2, interpolation='nearest')
            # plt.colorbar()
            # plt.show()
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dict_mode = {}
            for pix in spatial_dict:
                val = spatial_dict[pix]
                if np.isnan(val):
                    continue

                # if val > 5:
                if val > 0:
                    mode = 'alleviation'
                # elif val < -5:
                elif val < 0:
                    mode = 'excerbation'
                else:
                    mode = 'normal'
                    # raise ValueError
                spatial_dict_mode[pix] = mode
            key = f.replace('.tif', '')
            spatial_dict_all[key] = spatial_dict_mode
        df = T.spatial_dics_to_df(spatial_dict_all)
        df = Dataframe_func(df).df
        # T.print_head_n(df);exit()
        # T.print_head_n(df)
        ratio_dict = {}
        ELI_class_list = global_ELI_class_list
        flag = 0
        for drt in drt_list:
            for season in global_drought_timing_list:
                col_name = f'{drt}_{season}'
                df_season = df.dropna(subset=[col_name])
                for ELI_class in ELI_class_list:
                    df_ELI = df_season[df_season['ELI_class'] == ELI_class]
                    mode_list = T.get_df_unique_val_list(df_ELI, col_name)
                    # print(mode_list)
                    for mode in mode_list:
                        df_mode = df_ELI[df_ELI[col_name] == mode]
                        count = len(df_mode)
                        ratio = count / len(df_ELI)
                        # print(season, ELI_class, mode, count, f'{ratio:.3f}%')
                        ratio_dict[flag] = {
                            'season': season,
                            'drt': drt,
                            'ELI_class': ELI_class,
                            'mode': mode,
                            'count': count,
                            'ratio': f'{ratio*100:.3f}%'
                        }
                        flag += 1
        df_result = T.dic_to_df(ratio_dict, 'flag')
        T.print_head_n(df_result)
        outf = join(outdir, 'ratio')
        T.df_to_excel(df_result, outf)

        pass

    def plot_season_excerbation_alleviation_ratio(self):
        fpath = join(self.this_class_arr, 'season_excerbation_alleviation_ratio_statistic', 'ratio.xlsx')
        outdir = join(self.this_class_png, 'plot_season_excerbation_alleviation_ratio')
        T.mk_dir(outdir)
        df = pd.read_excel(fpath, index_col=0)
        T.print_head_n(df)
        # exit()
        drt_list = ['hot', 'normal']
        ELI_class_list = global_ELI_class_list
        mode_list = ['alleviation', 'excerbation']

        for ELI in ELI_class_list:
            plt.figure(figsize=(5, 5))
            df_ELI = df[df['ELI_class'] == ELI]
            for season in global_drought_timing_list:
                df_season = df_ELI[df_ELI['season'] == season]
                for drt in drt_list[::-1]:
                    df_drt = df_season[df_season['drt'] == drt]
                    for mode in mode_list:
                        df_mode = df_drt[df_drt['mode'] == mode]
                        print(df_mode)
                        if len(df_mode) != 1:
                            raise
                        ratio_str = df_mode['ratio'].tolist()[0]
                        ratio = ratio_str.split('%')[0]
                        ratio = float(ratio)
                        color = 'blue'
                        if mode == 'excerbation':
                            ratio = -ratio
                            color = 'red'
                        plt.bar(f'{season} {drt}', ratio, color=color)
                        # key = f'{season} {ELI} {drt} {mode}'
                        # bar_dict[key] = {'ratio':ratio,'color':color}
            # pprint(bar_dict)
            plt.title(f'{ELI}')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.ylim(-95, 60)
            outf = join(outdir, f'{ELI}.pdf')
            # plt.savefig(outf, dpi=300)
            # plt.close()
            plt.show()

    def plot_drought_events_timeseries_different_season(self,df):
        outdir = join(self.this_class_png, 'drought_events_timeseries_different_season')
        T.mk_dir(outdir)
        df = df.dropna(subset=['drought_season'])
        for AI_class in global_AI_class_list:
            df_AI = df[df['AI_class'] == AI_class]
            year_list = T.get_df_unique_val_list(df_AI,'drought_year')
            result_dict = {}
            df_group_dict = T.df_groupby(df_AI, 'drought_year')
            key_list = []
            for year in tqdm(df_group_dict):
                df_i = df_group_dict[year]
                result_dict_i = {}
                for season in global_drought_timing_list:
                    df_season = df_i[df_i['drought_season'] == season]
                    key = f'{season}'
                    if not key in key_list:
                        key_list.append(key)
                    count = len(df_season)
                    result_dict_i[key] = count
                result_dict[year] = result_dict_i
            df_result = T.dic_to_df(result_dict,'year')
            T.print_head_n(df_result)
            for timing in global_drought_timing_list:
                count_list = []
                for year in year_list:
                    df_year = df_result[df_result['year'] == year]
                    count = df_year[f'{timing}'].tolist()[0]
                    count_list.append(count)
                plt.plot(year_list, count_list, label=timing)
            plt.title(f'{AI_class}')
            plt.xlabel('year')
            plt.ylabel('count')
            plt.legend()
            plt.tight_layout()
            outf = join(outdir, f'{AI_class}.pdf')
            plt.savefig(outf, dpi=300)
            plt.close()

        pass

    def plot_spatial_trend_of_temperature_during_drought(self,df):
        outdir = join(self.this_class_tif, 'plot_spatial_trend_of_temperature_during_drought')
        T.mk_dir(outdir)
        drt_type_list = global_drought_type_list
        col = 'Temperature-anomaly_detrend'
        for drt in drt_type_list:
            df_drt = df[df['drought_type'] == drt]
            df_pix = T.df_groupby(df_drt, 'pix')
            spatial_dict = {}
            for pix in tqdm(df_pix, desc=drt):
                df_pix_i = df_pix[pix]
                drought_year = df_pix_i['drought_year'].tolist()
                temperature_list = df_pix_i[col].tolist()
                # plt.plot(drought_year, temperature_list, label=pix)
                # plt.show()
                # pause()
                try:
                    a,b,r,p = T.nan_line_fit(drought_year,temperature_list)
                    spatial_dict[pix] = a
                except:
                    pass
            outf = join(outdir, f'{drt}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict, outf)

        pass

    def plot_temporal_trend_of_temperature_during_drought(self,df):
        # outdir = join(self.this_class_tif, 'plot_spatial_trend_of_temperature_during_drought')
        # T.mk_dir(outdir)
        drt_type_list = global_drought_type_list
        col = 'Temperature-anomaly_detrend'
        for drt in drt_type_list:
            df_drt = df[df['drought_type'] == drt]
            df_year = T.df_groupby(df_drt, 'drought_year')
            year_list = []
            temperature_mean_list = []
            temperature_std_list = []
            for year in tqdm(df_year, desc=drt):
                df_year_i = df_year[year]
                temperature_list = df_year_i[col].tolist()
                temperature_mean = np.nanmean(temperature_list)
                temperature_std = np.nanstd(temperature_list)
                year_list.append(year)
                temperature_mean_list.append(temperature_mean)
                temperature_std_list.append(temperature_std)
            plt.figure()
            plt.errorbar(year_list, temperature_mean_list, yerr=temperature_std_list, label=drt)
            plt.title(f'{drt}')
            plt.xlabel('year')
            plt.ylabel('Temperature-anomaly_detrend')
            plt.legend()
        plt.show()
        # pause()


        pass

    def plot_SOS_NDVI_seasonal_time_series_during_drought(self,df):
        outdir = join(self.this_class_png, 'plot_SOS_NDVI_seasonal_time_series_during_drought')
        T.mk_dir(outdir)
        gs = np.arange(5,11)
        print(gs)
        AI_class_list = global_AI_class_list
        df = df.dropna(subset=['drought_season'])
        drought_type = 'normal-drought'
        # drought_type = 'hot-drought'
        # drought_type = 'all'
        df = df[df['drought_type'] == drought_type]
        for sos_type in [0,1]:
            if sos_type == 0:
                df_sos = df[df['SOS'] < 0]
                sos_type_str = 'advanced'
            else:
                df_sos = df[df['SOS'] > 0]
                sos_type_str = 'delayed'
            for season in global_drought_timing_list:
                df_season = df_sos[df_sos['drought_season'] == season]
                plt.figure(figsize=(2.0111, 1.3594))
                for AI_class in AI_class_list:
                    df_AI = df_season[df_season['AI_class'] == AI_class]
                    NDVI_vals = df_AI['NDVI-anomaly_detrend_drought_year_whole_year'].tolist()
                    NDVI_vals_mean = np.nanmean(NDVI_vals,axis=0)
                    NDVI_vals_std = np.nanstd(NDVI_vals,axis=0)/4.
                    # NDVI_vals_std = self.uncertainty_err_2d(NDVI_vals,axis=0)
                    # print(f'{sos_type}_{season}_{AI_class}')
                    # print(len(NDVI_vals))
                    plt.plot(NDVI_vals_mean,label=f'{AI_class}',marker='o',markersize=3)
                    plt.fill_between(range(len(NDVI_vals_mean)),NDVI_vals_mean-NDVI_vals_std,NDVI_vals_mean+NDVI_vals_std,alpha=0.5)
                plt.ylim(-1.2,.7)
                plt.hlines(0,0,11,colors='black',linestyles='--')
                # plt.legend()
                plt.title(f'{drought_type}_{sos_type_str}_{season}')
                # plt.grid()
                # plt.show()
                # exit()
                outf = join(outdir, f'{drought_type}_{sos_type_str}_{season}.pdf')
                plt.savefig(outf)
                plt.close()
                # exit()

        pass

    def plot_SOS_NDVI_seasonal_bar_during_drought(self,df):
        outdir = join(self.this_class_png, 'plot_SOS_NDVI_seasonal_bar_during_drought')
        T.mk_dir(outdir)
        AI_class_list = global_AI_class_list[::-1]
        df = df.dropna(subset=['drought_season'])
        drought_type = 'normal-drought'
        # drought_type = 'hot-drought'
        # drought_type = 'all'
        df = df[df['drought_type'] == drought_type]
        for sos_type in [0,1]:
            if sos_type == 0:
                df_sos = df[df['SOS'] < 0]
                sos_type_str = 'advanced'
            else:
                df_sos = df[df['SOS'] > 0]
                sos_type_str = 'delayed'
            x_ticks = []
            NDVI_vals_mean_list = []
            NDVI_vals_std_list = []
            plt.figure(figsize=(2.0111, 1.3594))
            for AI_class in AI_class_list:
                for season in global_drought_timing_list:
                    df_AI = df_sos[df_sos['AI_class'] == AI_class]
                    df_season = df_AI[df_AI['drought_season'] == season]

                    NDVI_vals = df_season['NDVI-anomaly_detrend_drought_year_whole_year'].tolist()
                    NDVI_vals_mean = np.nanmean(NDVI_vals)
                    # NDVI_vals_std = np.nanstd(NDVI_vals)
                    NDVI_vals = np.array(NDVI_vals)
                    NDVI_vals_flatten = NDVI_vals.flatten()
                    err, up, bottom = T.uncertainty_err(NDVI_vals_flatten)
                    err = abs(err)
                    # print(err, up, bottom)
                    # NDVI_vals_std = self.uncertainty_err_2d(NDVI_vals,axis=0)
                    # print(f'{sos_type}_{season}_{AI_class}')
                    # print(len(NDVI_vals))
                    x_ticks.append(f'{season} {AI_class}')
                    NDVI_vals_mean_list.append(NDVI_vals_mean)
                    NDVI_vals_std_list.append(err)
            plt.bar(x_ticks,NDVI_vals_mean_list,yerr=NDVI_vals_std_list)
            plt.title(f'{drought_type}_{sos_type_str}')
            plt.xticks(rotation=90)
            plt.ylim(-.41,.1)
            # plt.show()
            # pause()
            outf = join(outdir,f'{drought_type}_{sos_type_str}.pdf')
            plt.savefig(outf)
            plt.close()

        pass

    def plot_seasonal_drought_number_gradient(self,df):
        outdir = join(self.this_class_png,'plot_seasonal_drought_number_gradient')
        T.mk_dir(outdir)
        AI_class_list = global_AI_class_list[::-1]
        df = df.dropna(subset=['drought_season'])
        drought_type = 'normal-drought'
        # drought_type = 'hot-drought'
        # drought_type = 'all'
        df = df[df['drought_type'] == drought_type]
        # T.print_head_n(df)
        # exit()
        for sos_type in [0, 1]:
            if sos_type == 0:
                df_sos = df[df['SOS'] < 0]
                sos_type_str = 'advanced'
            else:
                df_sos = df[df['SOS'] > 0]
                sos_type_str = 'delayed'
            x_ticks = []
            for season in global_drought_timing_list:
                df_season = df_sos[df_sos['drought_season'] == season]
                season_range = df_season[f'{season}_range'].tolist()
                season_range_flatten = [item for sublist in season_range for item in sublist]
                season_range_flatten = np.array(season_range_flatten,dtype=float)
                # plt.hist(season_range_flatten,bins=12,range=(0,12))
                x,y = Plot().plot_hist_smooth(season_range_flatten,bins=12,range=(0,12),alpha=0.,interpolate_window=1)
                plt.close()
                x_interp = np.linspace(0,12,120)
                y_interp = np.interp(x_interp,x,y)
                y_interp_smooth = SMOOTH().smooth_convolve(y_interp,window_len=11)
                # plt.plot(x_interp,y_interp_smooth)
                # plt.show()
                # exit()
                y_interp_smooth = T.normalize(y_interp_smooth, norm_max=1., norm_min=0)
                arr = np.ones((len(x_interp),len(x_interp))) * y_interp_smooth
                plt.figure(figsize=(2.0111, 1.3594))
                # plt.imshow(arr,cmap='gray_r',vmin=0,vmax=2,zorder=0)
                rgba = plt.cm.gray_r(arr)
                rgba[...,-1] = arr
                plt.imshow(rgba,alpha=0.5,aspect='auto')
                plt.xlim(0,len(x_interp))
                plt.title(f'{drought_type}_{sos_type_str}_{season}')
                outf = join(outdir, f'{drought_type}_{sos_type_str}_{season}.pdf')
                plt.savefig(outf)
                plt.close()
        pass

    def plot_Temperature_vs_SOS(self,df):
        outdir = join(self.this_class_png,'plot_Temperature_vs_SOS')
        T.mk_dir(outdir)
        drt = 'hot-drought'
        # drt = 'normal-drought'
        # df = df[df['drought_type']==drt]
        df = df[df['SOS']<30]
        df = df[df['SOS']>-30]
        Temperature_col = 'Temperature-anomaly'
        sos_col = 'SOS'
        temp_bin = np.linspace(-1.5,1.5,31)
        df_group, bins_list_str = T.df_bin(df,Temperature_col,temp_bin)
        mean_list = []
        err_list = []
        x_list = []
        for name,df_group_i in df_group:
            x = name[0].left
            vals = df_group_i[sos_col].tolist()
            mean = np.nanmean(vals)
            err = np.nanstd(vals)
            mean_list.append(mean)
            err_list.append(err)
            x_list.append(x)
        x_list = np.array(x_list)
        mean_list = np.array(mean_list)
        err_list = np.array(err_list)
        plt.plot(x_list,mean_list,'o-')
        plt.fill_between(x_list,mean_list-err_list,mean_list+err_list,alpha=0.5)
        plt.xlim(-1.5,1.5)
        plt.ylim(-20,20)
        plt.show()
        # outf = join(outdir,f'{drt}.pdf')
        # plt.savefig(outf)
        # plt.close()
        pass

    def print_early_peak_late_reduction(self,df):
        drought_season_list = global_drought_timing_list
        for drought_season in drought_season_list:
            df_season = df[df['drought_season'] == drought_season]
            vals = df_season['NDVI-anomaly_detrend_drought_growing_season'].values
            vals_mean = np.nanmean(vals)
            print(f'{drought_season}: {vals_mean}')
        pass

    def plot_SOS_during_drought(self,df):
        outtif_dir = join(self.this_class_tif,'SOS_during_drought')
        outpng_dir = join(self.this_class_png,'SOS_during_drought')
        T.mk_dir(outtif_dir)
        T.mk_dir(outpng_dir)
        drt_list = global_drought_type_list
        for drt in drt_list:
            df_drt = df[df['drought_type'] == drt]
            pix_dict = T.df_groupby(df_drt,'pix')
            spatial_dict = {}
            for pix in pix_dict:
                df_pix = pix_dict[pix]
                SOS = df_pix['SOS'].tolist()
                SOS_mean = np.nanmean(SOS)
                spatial_dict[pix] = SOS_mean
            outf = join(outtif_dir,f'{drt}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict,outf)
            color_list = [
                '#075125',
                '#449364',
                '#ffffff',
                '#cc9999',
                '#79287a',
            ]
            cmap = T.cmap_blend(color_list)
            Plot().plot_ortho(outf, ax=None, cmap=cmap, vmin=-10, vmax=10, is_plot_colorbar=True, is_reproj=True,is_discrete=False,colormap_n=11)
            plt.title(drt)
            outpng = join(outpng_dir,f'{drt}.png')
            plt.savefig(outpng,dpi=1200)
            plt.close()

        pass

    def check_df(self,df):
        df_drop_dup = df.drop_duplicates(subset=['drought_year','drought_mon','pix'],)
        T.print_head_n(df_drop_dup)

        pass

    def GS_length_during_drought(self,df):
        outdir = join(self.this_class_tif,'GS_length_during_drought')
        T.mk_dir(outdir)
        spatial_dict = {}
        df_group = T.df_groupby(df,'pix')
        for pix in tqdm(df_group):
            df_i = df_group[pix]
            SOS_list = df_i['SOS'].tolist()
            EOS_list = df_i['EOS'].tolist()
            SOS_mean = np.nanmean(SOS_list)
            EOS_mean = np.nanmean(EOS_list)
            spatial_dict[pix] = EOS_mean - SOS_mean
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        outf = join(outdir,'all_drought.tif')
        DIC_and_TIF().arr_to_tif(arr,outf)
        drought_type_list = global_drought_type_list
        for drt in drought_type_list:
            df_drt = df[df['drought_type'] == drt]
            spatial_dict = {}
            df_group = T.df_groupby(df_drt,'pix')
            for pix in df_group:
                df_i = df_group[pix]
                SOS_list = df_i['SOS'].tolist()
                EOS_list = df_i['EOS'].tolist()
                SOS_mean = np.nanmean(SOS_list)
                EOS_mean = np.nanmean(EOS_list)
                spatial_dict[pix] = EOS_mean - SOS_mean
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
            outf = join(outdir,f'{drt}.tif')
            DIC_and_TIF().arr_to_tif(arr,outf)
        T.open_path_and_file(outdir)
        pass

    def plot_GS_length_during_drought(self):
        fdir = join(self.this_class_tif,'GS_length_during_drought')
        outdir = join(self.this_class_png,'GS_length_during_drought')
        T.mk_dir(outdir)
        drought_type_list = [
            'all_drought',
            'hot-drought',
            'normal-drought'
        ]
        for drt in drought_type_list:
            fpath = join(fdir,f'{drt}.tif')
            # arr = DIC_and_TIF().spatial_tif_to_arr(fpath)
            outf = join(outdir,f'{drt}.png')
            Plot().plot_ortho(fpath,ax=None,cmap='RdBu',vmin=-20,vmax=20,is_plot_colorbar=True,is_reproj=True,is_discrete=False)
            plt.title(drt)
            plt.savefig(outf,dpi=1200)
            plt.close()
        T.open_path_and_file(outdir)
        pass

    def GS_length_during_drought_vs_AI(self):
        fdir = join(self.this_class_tif,'GS_length_during_drought')
        outdir = join(self.this_class_png,'GS_length_during_drought_vs_AI')
        T.mk_dir(outdir)
        f = join(data_root, 'Aridity_Index/aridity_index.tif')
        spatial_dict_AI = DIC_and_TIF().spatial_tif_to_dic(f)
        drought_type_list = [
            'all_drought',
            'hot-drought',
            'normal-drought'
        ]
        all_dict = {}
        for drt in drought_type_list:
            fpath = join(fdir,f'{drt}.tif')
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            all_dict[drt] = spatial_dict
        df = T.spatial_dics_to_df(all_dict)
        df = T.add_spatial_dic_to_df(df, spatial_dict_AI, 'aridity_index')

        AI_bins = np.linspace(0.1,2.6,12)
        df_group, bins_list_str = T.df_bin(df,'aridity_index',AI_bins)
        for drt in drought_type_list:
            mean_list = []
            err_list = []
            for name,df_group_i in df_group:
                vals = df_group_i[drt].tolist()
                mean = np.nanmean(vals)
                err,_,_ = T.uncertainty_err(vals)
                mean_list.append(mean)
                err_list.append(err)
            plt.plot(AI_bins[1:],mean_list,label=drt)
            mean_list = np.array(mean_list)
            plt.fill_between(AI_bins[1:],mean_list-err_list,mean_list+err_list,alpha=0.5)
        plt.legend()
        # plt.show()
        outf = join(outdir,'all_drought.pdf')
        plt.savefig(outf)
        plt.close()
        T.open_path_and_file(outdir)

        pass

    def plot_AI_histogram(self,df):
        outdir = join(self.this_class_png,'AI_histogram')
        T.mk_dir(outdir)
        df_group = T.df_groupby(df,'pix')
        spatial_dict_AI = {}
        for pix in df_group:
            df_i = df_group[pix]
            AI_list = df_i['aridity_index']
            AI = np.nanmean(AI_list)
            spatial_dict_AI[pix] = AI
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_AI)
        arr_flatten = arr.flatten()
        plt.figure(figsize=(6.5,2))
        x,y = Plot().plot_hist_smooth(arr_flatten,range=(0.1,2.6),alpha=0,bins=50)
        plt.plot(x,y)
        outf = join(outdir,'AI_histogram.pdf')
        plt.savefig(outf)
        plt.close()
        T.open_path_and_file(outdir)


        pass

    def GS_length_during_drought_advanced_delayed_SOS(self,df):
        outdir = join(self.this_class_tif,'GS_length_during_drought_advanced_delayed_SOS')
        outdir_png = join(self.this_class_png,'GS_length_during_drought_advanced_delayed_SOS')

        # outdir = join(self.this_class_tif,'EOS_during_drought_advanced_delayed_SOS')
        # outdir_png = join(self.this_class_png,'EOS_during_drought_advanced_delayed_SOS')
        T.mk_dir(outdir)
        T.mk_dir(outdir_png)

        advanced_delay_list = ['advanced','delayed']

        for adv in advanced_delay_list:
            if adv == 'advanced':
                df_adv = df[df['SOS']<0]
            else:
                df_adv = df[df['SOS']>0]
            spatial_dict = {}
            df_group = T.df_groupby(df_adv,'pix')
            for pix in tqdm(df_group):
                df_i = df_group[pix]
                SOS_list = df_i['SOS'].tolist()
                EOS_list = df_i['EOS'].tolist()
                SOS_mean = np.nanmean(SOS_list)
                EOS_mean = np.nanmean(EOS_list)
                spatial_dict[pix] = EOS_mean - SOS_mean
                # spatial_dict[pix] = EOS_mean
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
            outf = join(outdir,f'{adv}_all_drought.tif')
            DIC_and_TIF().arr_to_tif(arr,outf)
            drought_type_list = global_drought_type_list
            for drt in drought_type_list:
                df_drt = df_adv[df_adv['drought_type'] == drt]
                spatial_dict = {}
                df_group = T.df_groupby(df_drt,'pix')
                for pix in df_group:
                    df_i = df_group[pix]
                    SOS_list = df_i['SOS'].tolist()
                    EOS_list = df_i['EOS'].tolist()
                    SOS_mean = np.nanmean(SOS_list)
                    EOS_mean = np.nanmean(EOS_list)
                    spatial_dict[pix] = EOS_mean - SOS_mean
                    # spatial_dict[pix] = EOS_mean
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
                outf = join(outdir,f'{adv}_{drt}.tif')
                DIC_and_TIF().arr_to_tif(arr,outf)
        # T.open_path_and_file(outdir)
        for f in T.listdir(outdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(outdir,f)
            Plot().plot_ortho(fpath,ax=None,cmap='RdBu',vmin=-20,vmax=20,is_plot_colorbar=True,is_reproj=True,is_discrete=False)
            # plt.title('EOS_'+f+'.png')
            plt.title('GS_'+f+'.png')
            plt.savefig(join(outdir_png,f),dpi=1200)
            plt.close()
        T.open_path_and_file(outdir_png)
        pass

    def uncertainty_err(self, vals):
        vals = np.array(vals)
        vals = vals[~np.isnan(vals)]
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        up, bottom = stats.t.interval(0.95, len(vals) - 1, loc=mean, scale=std / np.sqrt(len(vals)))
        err = mean - bottom
        return err, up, bottom

    def uncertainty_err_2d(self, vals, axis=0):
        vals = np.array(vals)
        if axis == 0:
            vals_T = vals.T
            vals_err = []
            for val in tqdm(vals_T, desc='uncertainty'):
                err, _, _ = self.uncertainty_err(val)
                vals_err.append(err)
            vals_err = np.array(vals_err)
        elif axis == 1:
            vals_T = vals
            vals_err = []
            for val in tqdm(vals_T, desc='uncertainty'):
                err, _, _ = self.uncertainty_err(val)
                vals_err.append(err)
            vals_err = np.array(vals_err)
        else:
            raise Exception('axis must be 0 or 1')
        return vals_err


    def copy_df(self):
        print('Warning: this function will overwrite the dataframe')
        print('Warning: this function will overwrite the dataframe')
        print('Warning: this function will overwrite the dataframe')
        pause()
        pause()
        dff = join(Drought_timing().this_class_arr,'Drought_timing.df')
        df = T.load_df(dff)
        T.save_df(df, self.dff)
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

class MAT_MAP:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('MAT_MAP', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe.df')
        pass

    def run(self):
        # self.copy_df()
        self.compensation_excerbation_MAT_MAP_matrix()
        # self.compensation_excerbation_MAT_MAP_scatter()
        # self.GEZ_MAT_MAP_scatter()
        # self.GEZ_AI_delta_NDVI()
        # self.GEZ_AI_anomaly_NDVI()
        # self.GEZ_MAT_anomaly_NDVI()
        # self.GEZ_MAT_MAP()
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
        dff = Drought_timing().dff
        df = T.load_df(dff)
        T.save_df(df,self.dff)
        T.df_to_excel(df, self.dff)

    def compensation_excerbation_MAT_MAP_matrix(self):
        outdir = join(self.this_class_png,'compensation_excerbation_MAT_MAP_matrix')
        T.mk_dir(outdir)
        compensation_excerbation_tif = join(Compensation_Excerbation().this_class_tif,'delta_hot_normal/drought_year_1.tif')
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
        plt.figure(figsize=(7, 6))
        col_name = 'compensation_excerbation'
        df_group_Topt_MAT_delta, bins_list_str_Topt_MAT_delta = T.df_bin(df,'MAT',bins_Topt_MAT_delta)
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
                plt.scatter(y_pos,x_pos,s=40,c=mean,vmin=-0.7,vmax=0.7,cmap='RdBu',marker='s',linewidths=0)
                # plt.scatter(x_pos,y_pos,s=13,c=mean,vmin=-0.01,vmax=0.01,cmap='RdBu',marker='s',linewidths=0)
        # plt.colorbar()
        plt.xlabel('MAT')
        plt.ylabel('MAP')
        plt.ylim(-100, 4500)
        plt.xlim(-13, 35)
        outf = join(outdir,'compensation_excerbation_MAT_MAP.pdf')
        plt.savefig(outf)
        # exit()
        pass


    def compensation_excerbation_MAT_MAP_scatter(self):
        outdir = join(self.this_class_png,'compensation_excerbation_MAT_MAP_scatter')
        T.mk_dir(outdir)
        compensation_excerbation_tif = join(Compensation_Excerbation().this_class_tif,'delta_hot_normal/drought_year_1.tif')
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
        # T.print_head_n(df)
        # exit()
        # df = df.dropna(how='any')
        # bins_Topt_MAT_delta = np.arange(-5,36,1)
        # bins_MAP = np.arange(0,4001,100)
        col_name = 'compensation_excerbation'
        col_MAT = 'MAT'
        col_MAP = 'MAP'
        temp_vals = df[col_MAT].tolist()
        precip_vals = df[col_MAP].tolist()
        color_list = df[col_name].tolist()

        plt.scatter(temp_vals,precip_vals,c=color_list,vmin=-.5,vmax=.5,cmap='RdBu',linewidths=0,alpha=0.5,s=4)
        # plt.scatter(x_pos,y_pos,s=13,c=mean,vmin=-0.01,vmax=0.01,cmap='RdBu',marker='s',linewidths=0)
        plt.colorbar()
        plt.xlabel('MAP')
        plt.ylabel('MAT')
        plt.ylim(0,4000)
        plt.xlim(-10,35)
        outf = join(outdir,'compensation_excerbation_MAT_MAP.pdf')
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

    def GEZ_MAT_MAP_scatter(self):
        outdir = join(self.this_class_png,'GEZ_MAT_MAP_scatter')
        T.mk_dir(outdir)
        # T.color_map_choice()
        df = self.__gen_df_init()
        # T.print_head_n(df)
        df = df.dropna(subset=['GEZ'])
        df = df.drop_duplicates(subset=['pix'],keep='first')
        # print(len(df));exit()
        gez_str_list = df['GEZ'].tolist()
        gez_list = T.get_df_unique_val_list(df,'GEZ')
        gez_list = list(gez_list)
        gez_code = list(range(len(gez_list)))
        gez_dict = T.dict_zip(gez_list,gez_code)
        # print(gez_list_shuffle);exit()
        gez_str_spatial_dict = T.df_to_spatial_dic(df,'GEZ')
        gez_code_spatial_dict = {}
        for pix in gez_str_spatial_dict:
            val = gez_str_spatial_dict[pix]
            val = gez_dict[val]
            gez_code_spatial_dict[pix] = val
        # Topt_MAT_delta_tif = join(self.this_class_tif,'mat_Topt_delta/Topt_MAT_delta.tif')
        # Topt_MAT_delta_dict = DIC_and_TIF().spatial_tif_to_dic(Topt_MAT_delta_tif)

        MAT_tif = join(data_root,r'CRU_tmp\mat','mat_gs.tif')
        MAT_dict = DIC_and_TIF().spatial_tif_to_dic(MAT_tif)

        MAP_tif = join(data_root,r'CRU_precip\map','map.tif')
        MAP_dict = DIC_and_TIF().spatial_tif_to_dic(MAP_tif)
        spatial_dict_all = {
            'gez_code':gez_code_spatial_dict,
            'gez_str':gez_str_spatial_dict,
            'MAT':MAT_dict,
            'MAP':MAP_dict
        }
        df = T.spatial_dics_to_df(spatial_dict_all)
        df = df.dropna(how='any')
        map = df['MAP'].tolist()
        col_name = 'gez_code'
        col_MAT = 'MAT'
        col_MAP = 'MAP'
        color_list = T.gen_colors(len(gez_list))
        flag = 0
        for gez in gez_list:
            df_gez = df[df['gez_str']==gez]
            temp_vals = df_gez[col_MAT].tolist()
            precip_vals = df_gez[col_MAP].tolist()

            plt.scatter(temp_vals,precip_vals,linewidths=0,alpha=1,s=8,label=gez,color=color_list[flag])

            flag += 1

            plt.xlabel('MAP')
            plt.ylabel('MAT')
            plt.ylim(0,2000)
            plt.xlim(-10,35)
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.show()
            # outf = join(outdir,f'{gez}.pdf')
            # plt.savefig(outf)
            # plt.close()


        pass

    def GEZ_AI_delta_NDVI(self):
        df = self.__gen_df_init()
        delta_tif = join(Drought_timing().this_class_tif,'delta/delta.tif')
        df_group_pix = T.df_groupby(df,'pix')
        GEZ_spatial_dict = {}
        AI_spatial_dict = {}
        for pix in df_group_pix:
            GEZ = df_group_pix[pix]['GEZ'].tolist()[0]
            AI = df_group_pix[pix]['aridity_index'].tolist()[0]
            GEZ_spatial_dict[pix] = GEZ
            AI_spatial_dict[pix] = AI
        delta_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(delta_tif)

        spatial_dicts = {
            'GEZ':GEZ_spatial_dict,
            'delta':delta_spatial_dict,
            'AI':AI_spatial_dict
        }

        df_new = T.spatial_dics_to_df(spatial_dicts)
        df_new = df_new.dropna()
        GEZ_list = T.get_df_unique_val_list(df_new,'GEZ')
        for GEZ in GEZ_list:
            df_GEZ = df_new[df_new['GEZ']==GEZ]
            if len(df_GEZ) < 100:
                continue
            AI = df_GEZ['AI'].tolist()
            RT = df_GEZ['delta'].tolist()
            AI_mean = np.nanmean(AI)
            RT_mean = np.nanmean(RT)
            AI_std = np.nanstd(AI)/4.
            RT_std = np.nanstd(RT)/4.
            plt.scatter(AI_mean,RT_mean)
            plt.errorbar(AI_mean,RT_mean,xerr=AI_std,yerr=RT_std)
            plt.text(AI_mean,RT_mean,GEZ)
            plt.xlabel('AI')
            plt.ylabel('NDVI delta')
        plt.show()
        pass

    def GEZ_AI_anomaly_NDVI(self):
        GEZ_color_dict = self.GEZ_color_dict()
        GEZ_marker_dict = self.GEZ_marker_dict()
        df = self.__gen_df_init()
        MAT_f = join(data_root, r"CRU_tmp\mat\mat_gs.tif")
        MAT_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(MAT_f)
        MAP_f = join(data_root, r"CRU_precip\map\map.tif")
        MAP_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(MAP_f)
        df = T.add_spatial_dic_to_df(df, MAT_spatial_dict, 'MAT')
        df = T.add_spatial_dic_to_df(df, MAP_spatial_dict, 'MAP')
        outdir = join(self.this_class_png,'GEZ_AI_anomaly_NDVI')
        T.mk_dir(outdir)
        drought_type = 'hot-drought'
        # drought_type = 'normal-drought'
        df = df[df['drought_type']==drought_type]
        plt.figure(figsize=(10,8))
        plt.title(f'{drought_type}')

        GEZ_list = T.get_df_unique_val_list(df,'GEZ')
        xx = []
        yy = []
        for GEZ in GEZ_list:
            if not GEZ in GEZ_color_dict:continue
            df_GEZ = df[df['GEZ']==GEZ]
            # if len(df_GEZ) < 500:
            #     continue
            AI = df_GEZ['aridity_index'].tolist()
            RT = df_GEZ['NDVI-anomaly_detrend'].tolist()
            AI_mean = np.nanmean(AI)
            RT_mean = np.nanmean(RT)
            AI_std = np.nanstd(AI)/2.
            RT_std = np.nanstd(RT)/2.
            plt.scatter(AI_mean,RT_mean,label=GEZ,color=GEZ_color_dict[GEZ],marker=GEZ_marker_dict[GEZ],
                        s=140,zorder=10,edgecolors='k')
            plt.errorbar(AI_mean,RT_mean,xerr=AI_std,yerr=RT_std,c='gray')
            # plt.text(AI_mean,RT_mean,GEZ)
            plt.xlabel('AI')
            plt.ylabel('NDVI-anomaly')
            xx.append(AI_mean)
            yy.append(RT_mean)
        # plt.legend(fontsize=10,ncol=2)
        sns.regplot(x=xx, y=yy, scatter=False, color='gray')
        plt.ylim(-0.8,0.3)
        plt.xlim(0,2)
        outf = join(outdir,f'{drought_type}.pdf')
        # plt.show()
        plt.savefig(outf)
        plt.close()
        pass

    def GEZ_color_dict(self):
        gez_list = ('Boreal coniferous forest', 'Boreal mountain system', 'Boreal tundra woodland', 'Polar', 'Subtropical desert', 'Subtropical dry forest', 'Subtropical humid forest', 'Subtropical mountain system', 'Subtropical steppe', 'Temperate continental forest', 'Temperate desert', 'Temperate mountain system', 'Temperate oceanic forest', 'Temperate steppe', 'Tropical desert', 'Tropical dry forest', 'Tropical moist forest', 'Tropical shrubland', 'Water')
        # for gez in gez_list:
        #     print(f"'{gez}':'#ff0000',")
            # exit()
        gez_color_dict = {
            'Boreal coniferous forest': '#F8A790',
            'Boreal mountain system': '#ACD481',
            'Boreal tundra woodland': '#E658A0',

            'Polar': '#454FA2',

            'Subtropical desert': '#F8A790',
            'Subtropical dry forest': '#86C9EF',
            'Subtropical humid forest': '#454FA2',
            'Subtropical mountain system': '#ACD481',
            'Subtropical steppe': '#E658A0',

            'Temperate continental forest': '#86C9EF',
            'Temperate desert': '#F8A790',
            'Temperate mountain system': '#ACD481',
            'Temperate oceanic forest': '#454FA2',
            'Temperate steppe': '#E658A0',

            'Tropical desert': '#F8A790',
        }
        return gez_color_dict
        pass

    def GEZ_marker_dict(self):
        marker_dict = {
            'Boreal coniferous forest': 'o',
            'Boreal mountain system': 'o',
            'Boreal tundra woodland': 'o',

            'Polar': '^',

            'Subtropical desert': 'D',
            'Subtropical dry forest': 'D',
            'Subtropical humid forest': 'D',
            'Subtropical mountain system': 'D',
            'Subtropical steppe': 'D',

            'Temperate continental forest': 'X',
            'Temperate desert': 'X',
            'Temperate mountain system': 'X',
            'Temperate oceanic forest': 'X',
            'Temperate steppe': 'X',

            'Tropical desert': '*',
        }
        return marker_dict

    def GEZ_MAT_MAP(self):
        GEZ_color_dict =self.GEZ_color_dict()
        df = self.__gen_df_init()
        outdir = join(self.this_class_png,'GEZ_MAT_MAP')
        T.mk_dir(outdir)
        MAT_f = join(data_root,r"CRU_tmp\mat\mat_gs.tif")
        MAT_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(MAT_f)
        MAP_f = join(data_root,r"CRU_precip\map\map.tif")
        MAP_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(MAP_f)
        df = T.add_spatial_dic_to_df(df, MAT_spatial_dict, 'MAT')
        df = T.add_spatial_dic_to_df(df, MAP_spatial_dict, 'MAP')

        GEZ_list = T.get_df_unique_val_list(df,'GEZ')
        # for GEZ in GEZ_list:
        #     print(GEZ)
        # exit()
        # print(GEZ_list);exit()
        plt.figure(figsize=(10, 8))
        for GEZ in GEZ_list:
            if not GEZ in GEZ_color_dict:continue
            df_GEZ = df[df['GEZ']==GEZ]
            if len(df_GEZ) < 100:
                continue
            # AI = df_GEZ['aridity_index'].tolist()
            MAP = df_GEZ['MAP'].tolist()
            RT = df_GEZ['NDVI-anomaly_detrend'].tolist()
            RT_mean = np.nanmean(RT)
            MAT = df_GEZ['MAT'].tolist()
            MAP_mean = np.nanmean(MAP)
            MAT_mean = np.nanmean(MAT)
            MAP_std = np.nanstd(MAP)/1.
            MAT_std = np.nanstd(MAT)/1.
            plt.scatter(MAT_mean,MAP_mean,c=GEZ_color_dict[GEZ],label=GEZ,s=140,zorder=99,
                        marker=self.GEZ_marker_dict()[GEZ],edgecolors='black')
            plt.errorbar(MAT_mean,MAP_mean,xerr=MAT_std,yerr=MAP_std,c='gray',zorder=-99)
            # plt.text(MAT_mean,MAP_mean,GEZ)
            # plt.xlabel('AI')
            plt.ylabel('MAP')
            # plt.ylabel('NDVI-anomaly')
            plt.xlabel('MAT')
            # plt.ylim(0,4000)
        # plt.colorbar()
        plt.legend()
        plt.show()
        # outf = join(outdir,f'GEZ_MAT_MAP.pdf')
        # plt.savefig(outf)
        # plt.close()
        pass

    def GEZ_MAT_anomaly_NDVI(self):
        GEZ_color_dict = self.GEZ_color_dict()
        GEZ_marker_dict = self.GEZ_marker_dict()
        df = self.__gen_df_init()
        MAT_f = join(data_root, r"CRU_tmp\mat\mat_gs.tif")
        MAT_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(MAT_f)
        MAP_f = join(data_root, r"CRU_precip\map\map.tif")
        MAP_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(MAP_f)
        df = T.add_spatial_dic_to_df(df, MAT_spatial_dict, 'MAT')
        df = T.add_spatial_dic_to_df(df, MAP_spatial_dict, 'MAP')
        outdir = join(self.this_class_png,'GEZ_MAT_anomaly_NDVI')
        T.mk_dir(outdir)
        # drought_type = 'hot-drought'
        drought_type = 'normal-drought'
        df = df[df['drought_type']==drought_type]
        plt.figure(figsize=(10,8))
        plt.title(f'{drought_type}')

        GEZ_list = T.get_df_unique_val_list(df,'GEZ')
        xx = []
        yy = []
        for GEZ in GEZ_list:
            if not GEZ in GEZ_color_dict:continue
            df_GEZ = df[df['GEZ']==GEZ]
            # if len(df_GEZ) < 500:
            #     continue
            # AI = df_GEZ['aridity_index'].tolist()
            AI = df_GEZ['MAT'].tolist()
            RT = df_GEZ['NDVI-anomaly_detrend'].tolist()
            AI_mean = np.nanmean(AI)
            RT_mean = np.nanmean(RT)
            AI_std = np.nanstd(AI)/2.
            RT_std = np.nanstd(RT)/2.
            plt.scatter(AI_mean,RT_mean,label=GEZ,c=GEZ_color_dict[GEZ],s=140,marker=GEZ_marker_dict[GEZ],
                        edgecolors='black',zorder=99)
            plt.errorbar(AI_mean,RT_mean,xerr=AI_std,yerr=RT_std,c='gray')
            xx.append(AI_mean)
            yy.append(RT_mean)
            # plt.text(AI_mean,RT_mean,GEZ)
            plt.xlabel('MAT')
            plt.ylabel('NDVI-anomaly')
        # plt.show()
        # plt.legend(fontsize=10,ncol=2)
        sns.regplot(x=xx, y=yy, scatter=False, color='gray')
        plt.ylim(-0.8,0.3)
        outf = join(outdir,f'{drought_type}.pdf')
        # plt.show()
        plt.savefig(outf)
        plt.close()
        pass

def Load_dataframe():
    dff = Dataframe().dff
    df = T.load_df(dff)
    return df

def main():
    # Dataframe().run()
    # Compensation_Excerbation().run()
    # Compensation_Excerbation_heatwave().run()
    # Drought_timing().run()
    Dynamic_gs_analysis().run()
    # Random_Forests().run()
    # Random_Forests_delta().run()
    # Partial_Dependence_Plots().run()
    # Partial_Dependence_Plots_Delta().run()
    # Single_Variable_Analysis().run()
    # Phenology_Statistic().run()
    # Optimal_temperature_statistic().run()
    # SEM().run()
    # MAT_MAP().run()

    pass

if __name__ == '__main__':
    main()
