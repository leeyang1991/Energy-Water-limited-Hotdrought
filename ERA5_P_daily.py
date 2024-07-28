# coding=utf-8
import matplotlib.pyplot as plt
from lytools import *
from pprint import pprint
T = Tools()
this_root = '/home/liyang/Desktop/disk/Precip_cv_trend/'
data_root = this_root + 'data/'
class ERA5_P:

    def __init__(self):
        self.datadir = join(data_root,'ERA5_P_daily')
        pass

    def run(self):
        # self.cal_cv()
        # self.cv_trend()
        # self.plot_cv_trend()
        # self.cal_extreme_events()
        # self.cal_extreme_events_trend()
        # self.plot_extreme_events_trend()
        # self.cal_dry_days()
        # self.cal_dry_days_trend()
        # self.plot_dry_days_trend()
        # self.composite_to_biweekly_26()
        self.composite_to_biweekly_24()

    def cal_cv(self):
        fdir = join(self.datadir,'precip_transform')
        outdir = join(self.datadir,'cv')
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            cv_spatial_dict = {}

            for pix in spatial_dict:
                vals = spatial_dict[pix]
                vals_reshape = np.reshape(vals,(-1,365))
                cv_list = []
                for year_vals in vals_reshape:
                    if T.is_all_nan(year_vals):
                        cv_list.append(np.nan)
                        continue
                    mean = np.nanmean(year_vals)
                    std = np.nanstd(year_vals)
                    cv = std/mean
                    cv_list.append(cv)
                cv_spatial_dict[pix] = cv_list
            outf = join(outdir,f)
            T.save_npy(cv_spatial_dict,outf)

    def cv_trend(self):
        fdir = join(self.datadir,'cv')
        outdir = join(self.datadir,'cv_trend')
        T.mk_dir(outdir,force=True)
        spatial_trend_dict = {}
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            for pix in spatial_dict:
                vals = spatial_dict[pix]
                # exit()
                if T.is_all_nan(vals):
                    continue

                slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(vals)),vals)
                spatial_trend_dict[pix] = {
                    'slope':slope,
                    'intercept':intercept,
                    'r_value':r_value,
                    'p_value':p_value,
                    'std_err':std_err
                }
        df = T.dic_to_df(spatial_trend_dict,'pix')
        outf = join(outdir,'trend.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def plot_cv_trend(self):
        dff = join(self.datadir,'cv_trend','trend.df')
        df = T.load_df(dff)
        spatial_dict_trend = T.df_to_spatial_dic(df,'slope')
        arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict_trend)
        plt.imshow(arr,cmap='RdBu',interpolation='nearest',vmin=-0.05,vmax=0.05)
        plt.colorbar()
        plt.show()
        pass


    def cal_extreme_events(self):
        fdir = join(self.datadir,'precip_transform')
        outdir = join(self.datadir,'extreme_events_num')
        T.mk_dir(outdir,force=True)
        spatial_dict_extreme_quantile = {}
        for f in tqdm(T.listdir(fdir)):
            # if not '005' in f:
            #     continue
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            extreme_event_spatial_dict = {}
            for pix in spatial_dict:
                vals = spatial_dict[pix]
                vals_reshape = np.reshape(vals,(-1,365))
                vals_flatten = vals_reshape.flatten()
                if T.is_all_nan(vals_flatten):
                    continue
                extreme_event_num_list = []
                quantile_90th = np.percentile(vals_flatten,90)
                for year_vals in vals_reshape:
                    if T.is_all_nan(year_vals):
                        extreme_event_num_list.append(np.nan)
                        continue
                    extreme_event = year_vals[year_vals>quantile_90th]
                    extreme_event_num = len(extreme_event)
                    # print(extreme_event)
                    extreme_event_num_list.append(extreme_event_num)
                extreme_event_spatial_dict[pix] = extreme_event_num_list

            outf = join(outdir,f)
            T.save_npy(extreme_event_spatial_dict,outf)
        # arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict_extreme_quantile)
        # plt.imshow(arr,cmap='RdBu',interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        # pass

    def cal_extreme_events_trend(self):
        fdir = join(self.datadir,'extreme_events_num')
        outdir = join(self.datadir,'extreme_events_trend')
        T.mk_dir(outdir,force=True)
        spatial_trend_dict = {}
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            for pix in spatial_dict:
                vals = spatial_dict[pix]
                slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(vals)), vals)
                spatial_trend_dict[pix] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_value': r_value,
                    'p_value': p_value,
                    'std_err': std_err
                }
        df = T.dic_to_df(spatial_trend_dict,'pix')
        outf = join(outdir,'trend.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def plot_extreme_events_trend(self):
        dff = join(self.datadir,'extreme_events_trend','trend.df')
        df = T.load_df(dff)
        spatial_dict_trend = T.df_to_spatial_dic(df,'slope')
        arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict_trend)
        plt.imshow(arr,cmap='RdBu',interpolation='nearest',vmin=-.6,vmax=.6)
        plt.colorbar()
        plt.title('extreme_wet_events_trend')
        plt.show()
        pass

    def cal_dry_days(self):
        fdir = join(self.datadir,'precip_transform')
        outdir = join(self.datadir,'dry_days')
        T.mk_dir(outdir,force=True)
        spatial_dict_extreme_quantile = {}
        for f in tqdm(T.listdir(fdir)):
            # if not '005' in f:
            #     continue
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            extreme_event_spatial_dict = {}
            for pix in spatial_dict:
                vals = spatial_dict[pix]
                vals_reshape = np.reshape(vals,(-1,365))
                vals_flatten = vals_reshape.flatten()
                if T.is_all_nan(vals_flatten):
                    continue
                extreme_event_num_list = []
                threshold = 1
                for year_vals in vals_reshape:
                    if T.is_all_nan(year_vals):
                        extreme_event_num_list.append(np.nan)
                        continue
                    extreme_event = year_vals[year_vals<threshold]
                    extreme_event_num = len(extreme_event)
                    # print(extreme_event)
                    extreme_event_num_list.append(extreme_event_num)
                extreme_event_spatial_dict[pix] = extreme_event_num_list

            outf = join(outdir,f)
            T.save_npy(extreme_event_spatial_dict,outf)
        # arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict_extreme_quantile)
        # plt.imshow(arr,cmap='RdBu',interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        # pass

    def cal_dry_days_trend(self):
        fdir = join(self.datadir,'dry_days')
        outdir = join(self.datadir,'dry_days_trend')
        T.mk_dir(outdir,force=True)
        spatial_trend_dict = {}
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            for pix in spatial_dict:
                vals = spatial_dict[pix]
                slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(vals)), vals)
                spatial_trend_dict[pix] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_value': r_value,
                    'p_value': p_value,
                    'std_err': std_err
                }
        df = T.dic_to_df(spatial_trend_dict,'pix')
        outf = join(outdir,'trend.df')
        T.save_df(df,outf)

    def plot_dry_days_trend(self):
        dff = join(self.datadir,'dry_days_trend','trend.df')
        df = T.load_df(dff)
        spatial_dict_trend = T.df_to_spatial_dic(df,'slope')
        arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict_trend)
        plt.imshow(arr,cmap='RdBu_r',interpolation='nearest',vmin=-.8,vmax=.8)
        # plt.imshow(arr,cmap='RdBu_r',interpolation='nearest')
        plt.colorbar()
        plt.title('dry_days_trend')
        plt.show()
        pass

    def composite_to_biweekly_26(self):
        fdir = join(self.datadir,'precip_transform')
        outdir = join(self.datadir,'biweekly_perpix_26')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            # if not '005' in f:
            #     continue
            outf = join(outdir,f)
            fpath = join(fdir,f)
            spatial_dict_i = T.load_npy(fpath)
            spatial_dict_biweekly = {}
            for pix in spatial_dict_i:
                vals = spatial_dict_i[pix]
                vals = np.array(vals)
                vals_flatten = vals.flatten()
                if T.is_all_nan(vals_flatten):
                    continue
                all_year_vals = []
                for daily_vals in vals:
                    # biweekly_vals = []
                    for i in range(0,len(daily_vals),14):
                        if not len(daily_vals[i:i+14])==14:
                            continue
                        all_year_vals.append(np.nansum(daily_vals[i:i+14]))
                all_year_vals = np.array(all_year_vals)
                spatial_dict_biweekly[pix] = all_year_vals
            T.save_npy(spatial_dict_biweekly,outf)
        pass

    def composite_to_biweekly_24(self):
        fdir = join(self.datadir,'precip_transform')
        outdir = join(self.datadir,'biweekly_perpix_24')
        T.mk_dir(outdir)
        date_list = []
        base_date = datetime.datetime(2001,1,1)
        for i in range(365):
            date_i = base_date + datetime.timedelta(days=i)
            date_list.append(date_i)
        date_group_dict = {}
        # for mon in range(1,13):

        for i, date in enumerate(date_list):
            mon = date.month
            day = date.day
            if day <= 15:
                if not (mon,1) in date_group_dict:
                    date_group_dict[(mon,1)] = []
                date_group_dict[(mon, 1)].append(i)
            else:
                if not (mon, 16) in date_group_dict:
                    date_group_dict[(mon, 16)] = []
                date_group_dict[(mon, 16)].append(i)

        for f in tqdm(T.listdir(fdir)):
            # if not '005' in f:
            #     continue
            outf = join(outdir,f)
            fpath = join(fdir,f)
            spatial_dict_i = T.load_npy(fpath)
            spatial_dict_biweekly = {}
            for pix in spatial_dict_i:
                vals = spatial_dict_i[pix]
                vals = np.array(vals)
                vals_flatten = vals.flatten()
                if T.is_all_nan(vals_flatten):
                    continue
                all_year_vals = []
                for year_val in vals:
                    for date_group in date_group_dict:
                        indx = date_group_dict[date_group]
                        vals_pick = year_val[indx]
                        vals_sum = np.nansum(vals_pick)
                        all_year_vals.append(vals_sum)
                all_year_vals = np.array(all_year_vals)
                spatial_dict_biweekly[pix] = all_year_vals
            T.save_npy(spatial_dict_biweekly,outf)
        pass

class CRU:

    def __init__(self):
        self.datadir = join(data_root,'CRU')
        pass

    def run(self):
        self.cal_cv()
        pass

    def cal_cv(self):
        fdir = join(self.datadir,'per_pix')
        spatial_dict = T.load_npy_dir(fdir)
        # T.moving_window_correlation()
        spatial_dict_cv_trend = {}
        window_size = 15
        date_list = None
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            if T.is_all_nan(vals):
                continue
            vals = np.array(vals)
            cv_list = []
            for i in range(len(vals)):
                if not date_list is None:
                    window_name = f'{date_list[i]}-{date_list[i + window_size]}'
                else:
                    window_name = f'{i}-{i + window_size}'
                picked_arr = vals[i:i + window_size]
                cv_i = np.std(picked_arr) / np.mean(picked_arr)
                cv_list.append(cv_i)
            slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(cv_list)), cv_list)

            spatial_dict_cv_trend[pix] = slope
        arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict_cv_trend)
        arr_flatten = arr.flatten()
        plt.hist(arr_flatten,bins=100)
        # plt.show()
        plt.figure()
        plt.imshow(arr,vmin=-0.01,vmax=0.01,cmap='jet',interpolation='nearest')
        plt.colorbar()
        plt.show()
        pass

def main():
    ERA5_P().run()
    # CRU().run()
    pass

if __name__ == '__main__':
    main()