# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

from meta_info import *
result_root_this_script = join(results_root, 'Supplementary')

class Different_VIS_analysis:

    def __init__(self):
        # self.VI_name = 'NDVI3g'
        # self.VI_name = 'NDVI4g'
        # self.VI_name = 'CSIF'
        self.VI_name = 'GPP_NIRv'
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            f'Different_VIS_analysis/{self.VI_name}',
            result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr,'dataframe.df')
        pass

    def run(self):
        self.copy_df()
        # ----------------------------
        df = self.__gen_df_init()
        df = self.add_GS_VI(df)
        df = self.add_percentage_process(df)

        T.save_df(df,self.dff)
        T.df_to_excel(df,self.dff)
        ############ ----------------------------
        self.tif_drought_year(df)
        self.delta_tif(df)
        self.Drought_year_percentage_spatial_tif(df)
        self.Drought_year_percentage_sig_spatial_tif()
        self.delta_percentage_sig()
        self.plot_delta_tif()

        self.plot_tif_drought_year()

        self.bar_percentage()
        self.percentage_delta_tif()
        self.delta_bar_percentage()
        self.anomaly_value_statistic()


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

    def add_GS_VI(self,df):
        if self.VI_name == 'NDVI3g':
            spatial_dict,var_name,valid_range = Load_Data().NDVI_3g_anomaly_detrend()
        elif self.VI_name == 'CSIF':
            spatial_dict,var_name,valid_range = Load_Data().CSIF_anomaly_detrend()
        elif self.VI_name == 'NDVI4g':
            spatial_dict,var_name,valid_range = Load_Data().NDVI_anomaly_detrend()
        elif self.VI_name == 'GPP_NIRv':
            spatial_dict,var_name,valid_range = Load_Data().GPP_NIRv_anomaly_detrend()
        else:
            raise IOError('VI_name error')
        year_list = year_range_str_to_list(global_VIs_year_range_dict[self.VI_name])
        # print(year_list)
        # exit()
        gs_mon_index = np.array(global_gs,dtype=int) - 1
        gs_vals_list = []
        for i,row in tqdm(df.iterrows(),total=len(df),desc=f'add_GS_{self.VI_name}'):
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
        df[self.VI_name+'_GS'] = gs_vals_list
        df = df[df[self.VI_name+'_GS']!=0]

        return df

    def add_percentage_process(self,df):
        # df = Load_dataframe()
        if self.VI_name == 'NDVI3g':
            NDVI_spatial_dict,var_name,valid_range = Load_Data().NDVI_3g_origin()
        elif self.VI_name == 'CSIF':
            NDVI_spatial_dict,var_name,valid_range = Load_Data().CSIF_origin()
        elif self.VI_name == 'NDVI4g':
            NDVI_spatial_dict,var_name,valid_range = Load_Data().NDVI_origin()
        elif self.VI_name == 'GPP_NIRv':
            NDVI_spatial_dict,var_name,valid_range = Load_Data().GPP_NIRv_origin()
        else:
            raise
        NDVI_percentage_spatial_dict = {}
        for pix in tqdm(NDVI_spatial_dict):
            NDVI = NDVI_spatial_dict[pix]
            NDVI[NDVI > valid_range[1]] = np.nan
            NDVI[NDVI < valid_range[0]] = np.nan
            if T.is_all_nan(NDVI):
                continue
            percentage = self.climatology_percentage(NDVI)
            NDVI_percentage_spatial_dict[pix] = percentage

        year_list = year_range_str_to_list(global_VIs_year_range_dict[self.VI_name])
        gs = global_gs
        NDVI_list_all = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            if not pix in NDVI_percentage_spatial_dict:
                NDVI_list_all.append(np.nan)
                continue
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
        df[f'{self.VI_name}_percentage_progress'] = NDVI_list_all
        return df

    def Drought_year_percentage_spatial_tif(self,df):
        outdir = join(self.this_class_tif,'Drought_year_percentage_spatial_tif')
        NDVI_data_dict = Load_Data().NDVI_origin()
        T.mk_dir(outdir,force=True)
        drought_type_list = global_drought_type_list
        for drt in drought_type_list:
            df_drt = df[df['drought_type']==drt]
            df_pix = T.df_groupby(df_drt,'pix')
            spatial_dict = {}
            for pix in tqdm(df_pix,desc=drt):
                df_pix_i = df_pix[pix]
                NDVI_progress = df_pix_i[f'{self.VI_name}_percentage_progress'].tolist()
                # plt.plot(NDVI_progress)
                # plt.show()
                try:
                    mean_progress = np.nanmean(NDVI_progress, axis=0)
                except:
                    continue
                try:
                    mean_progress_reshape = np.array(mean_progress).reshape(-1, 6)
                except:
                    continue
                if len(mean_progress_reshape) != 6:
                    continue
                # print(mean_progress_reshape)
                mean_progress_reshape_drought_year = mean_progress_reshape[1]
                mean_drought_year_NDVI = np.nanmean(mean_progress_reshape_drought_year)
                if mean_drought_year_NDVI > 50:
                    continue
                if mean_drought_year_NDVI < -50:
                    continue
                spatial_dict[pix] = mean_drought_year_NDVI
            outf = join(outdir,'{}.tif'.format(drt))
            DIC_and_TIF().pix_dic_to_tif(spatial_dict,outf)
        # T.open_path_and_file(outdir)

    def Drought_year_percentage_sig_spatial_tif(self):
        fdir = join(self.this_class_tif,'Drought_year_percentage_spatial_tif')
        outdir = join(self.this_class_tif,'Drought_year_percentage_sig_spatial_tif')
        T.mk_dir(outdir,force=True)
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            df = T.spatial_dics_to_df({'percentage':spatial_dict})

            sig_list = []
            for i,row in df.iterrows():
                val = row['percentage']
                if val > 5 or val < -5:
                    sig_list.append(0.0001)
                else:
                    sig_list.append(1)
            df['sig'] = sig_list
            spatial_dict_sig = T.df_to_spatial_dic(df,'sig')
            # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_sig)
            DIC_and_TIF().pix_dic_to_tif(spatial_dict_sig,outpath)


    def tif_drought_year(self,df):
        outdir = join(self.this_class_tif,'drought_year')
        T.mk_dir(outdir,force=True)
        drought_type_list = global_drought_type_list

        for drt in drought_type_list:
            df_drt = df[df['drought_type']==drt]
            spatial_dict = {}
            df_group_dict = T.df_groupby(df_drt,'pix')
            for pix in df_group_dict:
                df_pix = df_group_dict[pix]
                drought_year_value = df_pix[f'{self.VI_name}_GS'].tolist()
                drought_year_mean = np.nanmean(drought_year_value)
                spatial_dict[pix] = drought_year_mean
            outf = join(outdir,f'{drt}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict,outf)

    def plot_tif_drought_year(self):
        fdir = join(self.this_class_tif,'drought_year')
        sig_fdir = join(self.this_class_tif,'Drought_year_percentage_sig_spatial_tif')
        outdir = join(self.this_class_png,'drought_year')
        T.mk_dir(outdir,force=True)
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            sig_fpath = join(sig_fdir,f)
            outf = join(outdir,f.replace('.tif','.png'))
            m,ret = Plot().plot_ortho(fpath,vmin=-1,vmax=1,cmap='RdBu')
            Plot().plot_ortho_significance_scatter(m,sig_fpath,temp_root)
            # plt.show()
            plt.savefig(outf, dpi=300)
            plt.close()
        # T.open_path_and_file(outdir)
        pass

    def delta_tif(self,df):
        fdir = join(self.this_class_tif,'drought_year')
        outdir = join(self.this_class_tif,'delta')
        T.mk_dir(outdir,force=True)
        hot_tif = join(fdir,'hot-drought.tif')
        normal_tif = join(fdir,'normal-drought.tif')

        hot_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(hot_tif)
        normal_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(normal_tif)

        delta_dict = {}
        for pix in hot_spatial_dict:
            hot_val = hot_spatial_dict[pix]
            if not pix in normal_spatial_dict:
                continue
            normal_val = normal_spatial_dict[pix]
            delta = hot_val - normal_val
            delta_dict[pix] = delta
        outf = join(outdir,'delta.tif')
        DIC_and_TIF().pix_dic_to_tif(delta_dict,outf)

    def delta_percentage_sig(self):
        fdir = join(self.this_class_tif,'Drought_year_percentage_spatial_tif')
        outdir = join(self.this_class_tif,'delta_percentage_sig')
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

    def plot_delta_tif(self):
        fdir = join(self.this_class_tif,'delta')
        outdir = join(self.this_class_png, 'plot_delta_tif')
        T.mk_dir(outdir,force=True)
        sig_fpath = join(self.this_class_tif,'delta_percentage_sig','delta_percentage_sig.tif')

        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            outf = join(outdir,f.replace('.tif','.png'))
            m,ret = Plot().plot_ortho(fpath,vmin=-0.5,vmax=.5,cmap='Spectral')
            Plot().plot_ortho_significance_scatter(m,sig_fpath,temp_root)
            plt.savefig(outf,dpi=300)
            plt.close()
        # T.open_path_and_file(outdir)

    def percentage_delta_tif(self):
        fdir = join(self.this_class_tif,'Drought_year_percentage_spatial_tif')
        outdir = join(self.this_class_tif,'percentage_delta')
        T.mk_dir(outdir)
        hot_tif = join(fdir,'hot-drought.tif')
        normal_tif = join(fdir,'normal-drought.tif')

        hot_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(hot_tif)
        normal_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(normal_tif)

        delta_dict = {}
        for pix in hot_spatial_dict:
            hot_val = hot_spatial_dict[pix]
            if not pix in normal_spatial_dict:
                continue
            normal_val = normal_spatial_dict[pix]
            delta = hot_val - normal_val
            delta_dict[pix] = delta
        outf = join(outdir,'delta.tif')
        DIC_and_TIF().pix_dic_to_tif(delta_dict,outf)

        pass

    def bar_percentage(self):
        import statistic
        fdir = join(self.this_class_tif, 'Drought_year_percentage_spatial_tif')
        outdir = join(self.this_class_png, 'bar_percentage')
        T.mk_dir(outdir,force=True)
        spatial_dics = {}
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            if not f.endswith('.tif'):
                continue
            key = f.replace('.tif', '')
            spatial_dic = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dics[key] = spatial_dic
        df = T.spatial_dics_to_df(spatial_dics)
        df = statistic.Dataframe_func(df).df
        print('-----')
        T.print_head_n(df)
        # exit()
        ELI_class_list = global_ELI_class_list
        plt.figure(figsize=(15 * centimeter_factor, 6 * centimeter_factor))
        for key in spatial_dics:
            for ELI in ELI_class_list:
                df_ELI = df[df['ELI_class'] == ELI]
                df_ELI = df_ELI[df_ELI[key] > -30]
                df_ELI = df_ELI[df_ELI[key] < 30]
                vals = df_ELI[key].tolist()
                vals = np.array(vals)
                strong_positive_vals_count = len(vals[vals > 5])
                positive_vals_count = len(vals[vals > 0])
                negative_vals_count = len(vals[vals < 0])
                strong_negative_vals_count = len(vals[vals < -5])
                total_count = len(vals)

                strong_pos_ratio = strong_positive_vals_count / total_count * 100
                pos_ratio = positive_vals_count / total_count * 100
                neg_ratio = negative_vals_count / total_count * 100
                strong_neg_ratio = strong_negative_vals_count / total_count * 100

                plt.barh(f'{ELI}_{key}', strong_pos_ratio, color='g', zorder=1, linewidth=1, edgecolor='k',
                         height=.5)
                plt.barh(f'{ELI}_{key}', pos_ratio, color='none', zorder=0, linewidth=1, edgecolor='k', height=.5,
                         )
                plt.barh(f'{ELI}_{key}', pos_ratio, color='none', zorder=2, linewidth=1, edgecolor='k', height=.5)
                plt.barh(f'{ELI}_{key}', -neg_ratio, color='none', zorder=0, linewidth=1, edgecolor='k', height=.5,
                         )
                plt.barh(f'{ELI}_{key}', -neg_ratio, color='none', zorder=2, linewidth=1, edgecolor='k', height=.5)
                plt.barh(f'{ELI}_{key}', -strong_neg_ratio, color='r', zorder=1, linewidth=1, edgecolor='k',
                         height=.5)
                plt.text(pos_ratio, f'{ELI}_{key}', f'{pos_ratio:.1f} ({strong_pos_ratio:.1f})', ha='left',
                         va='center')
                plt.text(-neg_ratio, f'{ELI}_{key}', f'{neg_ratio:.1f} ({strong_neg_ratio:.1f})', ha='right',
                         va='center')
        plt.xlim(-150, 150)
        plt.tight_layout()
        # plt.show()
        outf = join(outdir, 'percentage.pdf')
        plt.savefig(outf)
        # T.open_path_and_file(outdir)

        pass

    def delta_bar_percentage(self):
        import statistic
        fdir = join(self.this_class_tif, 'percentage_delta')
        outdir = join(self.this_class_png, 'delta_bar_percentage')
        T.mk_dir(outdir)
        spatial_dics = {}
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            if not f.endswith('.tif'):
                continue
            key = f.replace('.tif', '')
            spatial_dic = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dics[key] = spatial_dic
        df = T.spatial_dics_to_df(spatial_dics)
        df = statistic.Dataframe_func(df).df
        print('-----')
        T.print_head_n(df)
        # exit()
        ELI_class_list = global_ELI_class_list
        plt.figure(figsize=(15 * centimeter_factor, 6 * centimeter_factor))
        for key in spatial_dics:
            for ELI in ELI_class_list:
                df_ELI = df[df['ELI_class'] == ELI]
                df_ELI = df_ELI[df_ELI[key] > -30]
                df_ELI = df_ELI[df_ELI[key] < 30]
                vals = df_ELI[key].tolist()
                vals = np.array(vals)
                strong_positive_vals_count = len(vals[vals > 5])
                positive_vals_count = len(vals[vals > 0])
                negative_vals_count = len(vals[vals < 0])
                strong_negative_vals_count = len(vals[vals < -5])
                total_count = len(vals)

                strong_pos_ratio = strong_positive_vals_count / total_count * 100
                pos_ratio = positive_vals_count / total_count * 100
                neg_ratio = negative_vals_count / total_count * 100
                strong_neg_ratio = strong_negative_vals_count / total_count * 100

                plt.barh(f'{ELI}_{key}', strong_pos_ratio, color='g', zorder=1, linewidth=1, edgecolor='k',
                         height=.5)
                plt.barh(f'{ELI}_{key}', pos_ratio, color='none', zorder=0, linewidth=1, edgecolor='k', height=.5,
                         )
                plt.barh(f'{ELI}_{key}', pos_ratio, color='none', zorder=2, linewidth=1, edgecolor='k', height=.5)
                plt.barh(f'{ELI}_{key}', -neg_ratio, color='none', zorder=0, linewidth=1, edgecolor='k', height=.5,
                         )
                plt.barh(f'{ELI}_{key}', -neg_ratio, color='none', zorder=2, linewidth=1, edgecolor='k', height=.5)
                plt.barh(f'{ELI}_{key}', -strong_neg_ratio, color='r', zorder=1, linewidth=1, edgecolor='k',
                         height=.5)
                plt.text(pos_ratio, f'{ELI}_{key}', f'{pos_ratio:.1f} ({strong_pos_ratio:.1f})', ha='left',
                         va='center')
                plt.text(-neg_ratio, f'{ELI}_{key}', f'{neg_ratio:.1f} ({strong_neg_ratio:.1f})', ha='right',
                         va='center')
        plt.xlim(-150, 150)
        plt.tight_layout()
        # plt.show()
        outf = join(outdir, 'percentage.pdf')
        plt.savefig(outf)
        # T.open_path_and_file(outdir)
        pass

    def anomaly_value_statistic(self):
        import statistic
        outdir = join(self.this_class_png, 'anomaly_value_statistic')
        T.mk_dir(outdir)
        outf = join(outdir,'delta_value_statistic.pdf')
        delta_tif = join(self.this_class_tif, 'delta/delta.tif')
        hot_tif = join(self.this_class_tif, 'drought_year/hot-drought.tif')
        normal_tif = join(self.this_class_tif,'drought_year/normal-drought.tif')

        delta_dict = DIC_and_TIF().spatial_tif_to_dic(delta_tif)
        hot_dict = DIC_and_TIF().spatial_tif_to_dic(hot_tif)
        normal_dict = DIC_and_TIF().spatial_tif_to_dic(normal_tif)

        all_spatial_dict = {
            'delta':delta_dict,
            'hot-drought':hot_dict,
            'normal-drought':normal_dict
        }
        df = T.spatial_dics_to_df(all_spatial_dict)
        df = statistic.Dataframe_func(df).df
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
        plt.hlines(0,-1,2,linestyles='--',colors='k')
        plt.ylim(-0.5,.2)
        plt.title('delta')

        plt.subplot(1,3,2)
        hot_mean_list = df_result['hot_mean'].tolist()
        hot_err_list = df_result['hot_err'].tolist()
        plt.bar(ELI_class_list_1,hot_mean_list,yerr=hot_err_list,width=0.5)
        plt.hlines(0,-1,2,linestyles='--',colors='k')
        plt.ylim(-0.5,.2)
        plt.title('hot-drought')

        plt.subplot(1,3,3)
        normal_mean_list = df_result['normal_mean'].tolist()
        normal_err_list = df_result['normal_err'].tolist()
        plt.bar(ELI_class_list_1,normal_mean_list,yerr=normal_err_list,width=0.5)
        plt.hlines(0,-1,2,linestyles='--',colors='k')
        plt.ylim(-0.5,.2)
        plt.title('normal-drought')

        plt.tight_layout()

        # plt.show()

        plt.savefig(outf,dpi=300)
        plt.close()
        # T.open_path_and_file(outdir)


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
        self.spi_scale_list = global_selected_spi_list
        pass

    def run(self):
        # self.copy_df()
        # ----------------------------
        # df = self.__gen_df_init()
        # df = self.add_GS_NDVI4g(df)
        # T.save_df(df,self.dff)
        # T.df_to_excel(df,self.dff)

        # self.tif_drought_year(df)
        # self.plot_tif_drought_year()
        # self.delta_tif()
        self.plot_delta_tif()
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
        # T.open_path_and_file(outdir)
        pass

    def delta_tif(self):
        fdir = join(self.this_class_tif,'drought_year')
        outdir = join(self.this_class_tif,'delta')
        T.mk_dir(outdir,force=True)
        spi_scale_list = global_selected_spi_list
        for scale in spi_scale_list:
            hot_tif = join(fdir,f'{scale}_hot-drought.tif')
            normal_tif = join(fdir,f'{scale}_normal-drought.tif')
            hot_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(hot_tif)
            normal_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(normal_tif)

            delta_dict = {}
            for pix in hot_spatial_dict:
                hot_val = hot_spatial_dict[pix]
                if not pix in normal_spatial_dict:
                    continue
                normal_val = normal_spatial_dict[pix]
                delta = hot_val - normal_val
                delta_dict[pix] = delta
            outf = join(outdir,f'{scale}_delta.tif')
            DIC_and_TIF().pix_dic_to_tif(delta_dict,outf)

    def plot_delta_tif(self):
        fdir = join(self.this_class_tif,'delta')
        outdir = join(self.this_class_png, 'plot_delta_tif')
        T.mk_dir(outdir,force=True)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            outf = join(outdir,f.replace('.tif','.png'))
            m,ret = Plot().plot_ortho(fpath,vmin=-0.5,vmax=.5,cmap='Spectral')
            plt.title(f.replace('.tif',''))
            plt.savefig(outf,dpi=300)
            plt.close()
        # T.open_path_and_file(outdir)

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
    Different_VIS_analysis().run()
    # Multi_SPI_scale_analysis().run()
    pass

if __name__ == '__main__':
    main()