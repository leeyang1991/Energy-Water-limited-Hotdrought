# coding=utf-8
import matplotlib.pyplot as plt

from meta_info import *
result_root_this_script = join(results_root, 'statistic_number')

class Figure1:
    def __init__(self):
        pass

    def run(self):
        # self.affected_area()
        self.affected_area_sig_CDH_all_area()
        # self.affected_area_high_lat()
        pass

    def affected_area(self):
        import statistic
        #/media/yang/HDD/Energy_water_hotdrought/results/statistic/Compensation_Excerbation/tif/Drought_year_spatial_tif
        fdir = join(statistic.Compensation_Excerbation().this_class_tif,'Drought_year_spatial_tif')
        spatial_dicts = {}
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            spatial_dic = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dicts[f.replace('.tif','')] = spatial_dic
        df = T.spatial_dics_to_df(spatial_dicts)
        # T.print_head_n(df)
        area_dict = DIC_and_TIF().calculate_pixel_area()
        T.add_spatial_dic_to_df(df, area_dict, 'area')
        # T.print_head_n(df)
        df_hot = df.dropna(subset=['hot-drought'])
        df_hot_neg = df_hot[df_hot['hot-drought'] <= -0.]
        area_hot = df_hot['area'].sum()
        area_hot_neg = df_hot_neg['area'].sum()
        hot_ratio = area_hot_neg / area_hot * 100

        df_normal = df.dropna(subset=['normal-drought'])
        df_normal_neg = df_normal[df_normal['normal-drought'] <= -0.]
        area_normal = df_normal['area'].sum()
        area_normal_neg = df_normal_neg['area'].sum()
        normal_ratio = area_normal_neg / area_normal * 100

        print('hot_ratio',hot_ratio)
        print('normal_ratio',normal_ratio)

        pass

    def affected_area_sig_CDH_all_area(self):
        # discussion first paragraph
        import statistic
        #/media/yang/HDD/Energy_water_hotdrought/results/statistic/Compensation_Excerbation/tif/Drought_year_spatial_tif
        fdir = join(statistic.Compensation_Excerbation().this_class_tif,'Drought_year_spatial_tif')
        # sig_fdir = '/mnt/ata-ST4000DM004-2CV104_WFN26VAE-part1/Energy_water_hotdrought/results/statistic/Compensation_Excerbation/tif/Drought_year_NDVI_percentage_sig_spatial_tif'
        sig_fdir = join(statistic.Compensation_Excerbation().this_class_tif,'Drought_year_NDVI_percentage_sig_spatial_tif')
        # T.open_path_and_file(sig_fdir);exit()
        spatial_dicts = {}
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            spatial_dic = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dicts[f.replace('.tif','')] = spatial_dic
        for f in T.listdir(sig_fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(sig_fdir,f)
            spatial_dic = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dicts[f.replace('.tif','')+'_sig'] = spatial_dic
        df = T.spatial_dics_to_df(spatial_dicts)
        # T.print_head_n(df);exit()
        area_dict = DIC_and_TIF().calculate_pixel_area()
        T.add_spatial_dic_to_df(df, area_dict, 'area')
        df[df['hot-drought_sig'] == 1] = np.nan
        T.print_head_n(df)

        df_hot = df.dropna(subset=['hot-drought'])
        df_hot_sig = df_hot.dropna(subset=['hot-drought_sig'])
        df_hot_neg = df_hot_sig[df_hot_sig['hot-drought'] <= -0.]
        area_all = df['area'].sum()
        area_hot_neg = df_hot_neg['area'].sum()
        hot_ratio = area_hot_neg / area_all * 100
        print('hot_ratio_sig',hot_ratio)
        hot_spatial_dict = T.df_to_spatial_dic(df_hot_neg,'hot-drought')
        # hot_spatial_dict = T.df_to_spatial_dic(df_hot,'hot-drought')
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(hot_spatial_dict)
        plt.imshow(arr,interpolation='nearest',cmap='RdBu_r',vmin=-99,vmax=-10)
        plt.colorbar()
        plt.show()


        pass

    def affected_area_high_lat(self):
        import statistic
        #/media/yang/HDD/Energy_water_hotdrought/results/statistic/Compensation_Excerbation/tif/Drought_year_spatial_tif
        fdir = join(statistic.Compensation_Excerbation().this_class_tif,'Drought_year_spatial_tif')
        # T.open_path_and_file(fdir);exit()
        spatial_dicts = {}
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            spatial_dic = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dicts[f.replace('.tif','')] = spatial_dic
        df = T.spatial_dics_to_df(spatial_dicts)
        # T.print_head_n(df)
        area_dict = DIC_and_TIF().calculate_pixel_area()
        df = T.add_spatial_dic_to_df(df, area_dict, 'area')
        df = T.add_lon_lat_to_df(df, DIC_and_TIF())
        df = df[df['lat'] > 60]
        df_hot = df.dropna(subset=['hot-drought'])
        df_hot_pos = df_hot[df_hot['hot-drought'] >= 0.]
        hot_spatial_dict = T.df_to_spatial_dic(df_hot,'hot-drought')
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(hot_spatial_dict)
        plt.imshow(arr,interpolation='nearest',vmin=-1,vmax=1,cmap='RdBu')
        arr_flatten = arr.flatten()
        arr_flatten = arr_flatten[~np.isnan(arr_flatten)]
        arr_flatten_pos = arr_flatten[arr_flatten > 0]
        pos_ratio = len(arr_flatten_pos) / len(arr_flatten) * 100
        print(pos_ratio)
        average_val = np.nanmean(arr_flatten)
        print('average_val',average_val)
        std_val = np.nanstd(arr_flatten)
        print('std_val',std_val)

        # T.print_head_n(df_hot_pos)
        # ratio = df_hot_pos['area'].sum() / df['area'].sum() * 100
        # print(ratio)

        pass

def main():
    Figure1().run()
    pass

if __name__ == '__main__':
    main()