# coding=utf-8
import copy

import matplotlib.pyplot as plt

from meta_info import *
result_root_this_script = join(results_root, 'presentation')

class NDVI_response_to_Temperature():

    def __init__(self):
        pass

class Spatial_figures:

    def __init__(self):
        import statistic as statistic
        self.statistic_class = statistic
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Spatial_figures', result_root_this_script, mode=2)
        pass

    def run(self):
        # self.gen_arid_humid_tif()
        # self.gen_arid_humid_tif_percentage_sig()
        self.plot_rt_tif()
        pass

    def gen_arid_humid_tif(self):
        dff = join(self.statistic_class.Compensation_Excerbation().dff)
        df = T.load_df(dff)
        outdir = join(self.this_class_tif, 'Drought_year_spatial_tif')
        T.mk_dir(outdir)
        AI_class_list = global_AI_class_list
        for AI_class in AI_class_list:
            df_AI = df[df['AI_class'] == AI_class]
            drought_type_list = global_drought_type_list
            for drt in drought_type_list:
                df_drt = df_AI[df_AI['drought_type'] == drt]
                df_pix = T.df_groupby(df_drt, 'pix')
                spatial_dict = {}
                for pix in tqdm(df_pix, desc=drt):
                    df_pix_i = df_pix[pix]
                    # T.print_head_n(df_pix_i);exit()
                    NDVI_progress = df_pix_i['NDVI_progress'].tolist()
                    mean_progress = np.nanmean(NDVI_progress, axis=0)
                    mean_progress_reshape = np.array(mean_progress).reshape(-1, 6)
                    mean_progress_reshape_drought_year = mean_progress_reshape[1]
                    mean_drought_year_NDVI = np.nanmean(mean_progress_reshape_drought_year)
                    spatial_dict[pix] = mean_drought_year_NDVI
                outf = join(outdir, '{}.tif'.format(drt))
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
                outf = join(outdir, f'{AI_class}_{drt}.tif')
                DIC_and_TIF().pix_dic_to_tif(spatial_dict, outf)
        pass

    def gen_arid_humid_tif_percentage_sig(self):
        fdir = join(self.statistic_class.Compensation_Excerbation().this_class_tif,'Drought_year_NDVI_percentage_sig_spatial_tif')
        outdir = join(self.this_class_tif,'delta_percentage_sig')
        T.mk_dir(outdir)
        fpath_hot = join(fdir,'hot-drought.tif')
        fpath_normal = join(fdir,'normal-drought.tif')
        spatial_dic_hot = DIC_and_TIF().spatial_tif_to_dic(fpath_hot)
        spatial_dic_normal = DIC_and_TIF().spatial_tif_to_dic(fpath_normal)
        spatial_dicts = {
            'hot-drought': spatial_dic_hot,
            'normal-drought': spatial_dic_normal
        }
        df = T.spatial_dics_to_df(spatial_dicts)
        df = self.statistic_class.Dataframe_func(df).df
        T.print_head_n(df)
        for drt in global_drought_type_list:
            df_drt = df.dropna(subset=[drt])
            for AI_class in global_AI_class_list:
                df_AI = df_drt[df_drt['AI_class'] == AI_class]
                df_pix = T.df_groupby(df_AI, 'pix')
                spatial_dict = {}
                for pix in tqdm(df_pix, desc=f'{AI_class}_{drt}'):
                    df_pix_i = df_pix[pix]
                    # T.print_head_n(df_pix_i);exit()
                    vals = df_pix_i[drt].tolist()[0]
                    spatial_dict[pix] = vals
                outf = join(outdir, f'{AI_class}_{drt}.tif')
                DIC_and_TIF().pix_dic_to_tif(spatial_dict, outf)
        pass

    def plot_arid_humid_tif(self):
        # self.statistic_class
        fdir = join(self.this_class_tif, 'Drought_year_spatial_tif')
        fdir_sig = join(self.this_class_tif, 'Drought_year_NDVI_percentage_sig_spatial_tif')

        outdir = join(self.this_class_png, 'Drought_year_spatial_tif')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            fpath_sig = join(fdir_sig, f)
            if not f.endswith('.tif'):
                continue
            outpath = join(outdir, f'{f.replace(".tif", "")}.png')
            m, ret = Plot().plot_ortho(fpath, vmin=-1, vmax=1, cmap='RdBu')
            Plot().plot_ortho_significance_scatter(m, fpath_sig, temp_root)

            plt.savefig(outpath, dpi=300)
            # plt.show()
            plt.close()
        T.open_path_and_file(outdir)
        pass

    def plot_rt_tif(self):
        fdir = join(self.this_class_tif,'Drought_year_spatial_tif')
        fdir_sig = join(self.this_class_tif,'delta_percentage_sig')

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


class Aridity_index:
    def __init__(self):
        self.datadir = join(data_root, 'Aridity_Index')

        pass

    def run(self):
        # self.gen_arid_humid_tif()
        # self.plot_ortho_tif_arid()
        self.plot_ortho_tif_humid()

        pass

    def gen_arid_humid_tif(self):
        fpath = join(self.datadir, 'aridity_index_binary.tif')
        arr = DIC_and_TIF().spatial_tif_to_arr(fpath)
        humid_arr = copy.copy(arr)
        humid_arr[arr == 0] = np.nan
        arid_arr = copy.copy(arr)
        arid_arr[arr == 1] = np.nan
        fpath_humid = join(self.datadir, 'aridity_index_humid.tif')
        fpath_arid = join(self.datadir, 'aridity_index_arid.tif')
        DIC_and_TIF().arr_to_tif(humid_arr, fpath_humid)
        DIC_and_TIF().arr_to_tif(arid_arr, fpath_arid)
        pass

    def plot_ortho_tif_arid(self):
        fpath = join(self.datadir, 'aridity_index_arid.tif')
        Plot().plot_ortho(fpath, cmap='RdBu', vmin=-.3, vmax=1.3)
        outf = join(self.datadir, 'aridity_index_arid.png')
        plt.savefig(outf, dpi=300)
        plt.close()
        pass

    def plot_ortho_tif_humid(self):
        fpath = join(self.datadir, 'aridity_index_humid.tif')
        Plot().plot_ortho(fpath, cmap='RdBu', vmin=-.3, vmax=1.3)
        outf = join(self.datadir, 'aridity_index_humid.png')
        plt.savefig(outf, dpi=300)
        plt.close()
        pass

def main():
    # Spatial_figures().run()
    Aridity_index().run()
    pass


if __name__ == '__main__':
    main()