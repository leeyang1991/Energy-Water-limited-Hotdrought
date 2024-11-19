# coding=utf-8
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

from __init__ import *
import xarray as xr
# import climate_indices
# from climate_indices import compute
# from climate_indices import indices
from meta_info import *

class GIMMS_NDVI:

    def __init__(self):
        self.datadir = join(data_root, 'NDVI4g')
        pass

    def run(self):
        # self.resample()
        # self.monthly_compose()
        # self.per_pix()
        # self.per_pix_clean()
        # self.per_pix_biweekly()
        # self.check_per_pix_biweekly()
        # self.per_pix_anomaly()
        # self.per_pix_anomaly_detrend()
        # self.per_pix_anomaly_detrend_GS()
        self.per_pix_percentage()
        pass

    def resample(self):
        fdir = join(self.datadir,'tif_8km_bi_weekly')
        outdir = join(self.datadir,'bi_weekly_05')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            ToRaster().resample_reproj(fpath,outpath,0.5)

    def monthly_compose(self):
        fdir = join(self.datadir,'bi_weekly_05')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir)
        Pre_Process().monthly_compose(fdir,outdir,method='max')
        pass

    def per_pix(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'per_pix')
        T.mk_dir(outdir)
        Pre_Process().data_transform(fdir,outdir)

    def per_pix_clean(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'per_pix_clean',global_year_range)
        T.mk_dir(outdir,force=True)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_clean = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            vals = np.array(vals)
            vals[vals>10000] = np.nan
            vals[vals<0] = np.nan
            vals = vals * 0.0001
            if T.is_all_nan(vals):
                continue
            spatial_dict_clean[pix] = vals
        outf = join(outdir,'per_pix_clean')
        T.save_npy(spatial_dict_clean,outf)


    def per_pix_biweekly(self):
        fdir = join(self.datadir,'bi_weekly_05')
        outdir = join(self.datadir,'per_pix_biweekly')
        T.mk_dir(outdir)
        Pre_Process().data_transform(fdir,outdir)

    def check_per_pix_biweekly(self):
        fdir = join(self.datadir,'per_pix_biweekly')
        # T.open_path_and_file(fdir)
        spatial_dict = T.load_npy_dir(fdir)
        for pix in spatial_dict:
            # print(pix)
            vals = spatial_dict[pix]
            vals = np.array(vals)
            vals[vals>10000] = np.nan
            # vals[vals<0] = np.nan
            if T.is_all_nan(vals):
                continue
            vals_reshape = np.reshape(vals,(-1,24))
            print(len(vals_reshape))
            plt.imshow(vals_reshape,cmap='RdBu',vmin=0,vmax=10000)
            plt.colorbar()

            plt.show()

    def per_pix_anomaly(self):
        fdir = join(self.datadir,'per_pix_clean',global_year_range)
        outdir = join(self.datadir,'per_pix_anomaly',global_year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)

    def per_pix_anomaly_detrend(self):
        fdir = join(self.datadir,'per_pix_anomaly',global_year_range)
        outdir = join(self.datadir,'per_pix_anomaly_detrend',global_year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().detrend(fdir,outdir)
        pass

    def per_pix_anomaly_detrend_GS(self):
        fdir = join(self.datadir,'per_pix_anomaly_detrend',global_year_range)
        outdir = join(self.datadir,'per_pix_anomaly_detrend_GS',global_year_range)
        T.mk_dir(outdir,force=True)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_gs = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            gs = global_gs
            vals_gs = T.monthly_vals_to_annual_val(vals,gs)
            spatial_dict_gs[pix] = vals_gs
        outf = join(outdir,'GS_mean.npy')
        T.save_npy(spatial_dict_gs,outf)

    def per_pix_percentage(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'per_pix_percentage',global_year_range)
        T.mk_dir(outdir,force=True)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_percentage = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            vals_percentage = self.climatology_percentage(vals)
            spatial_dict_percentage[pix] = vals_percentage
        outf = join(outdir,'vals_percentage.npy')
        T.save_npy(spatial_dict_percentage,outf)

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


class SPEI:

    def __init__(self):
        self.datadir = join(data_root, 'SPEI')
        pass

    def run(self):
        # self.nc_to_tif()
        # self.per_pix()
        # self.clean()
        # self.every_month()
        self.pick_year_range()
        pass

    def nc_to_tif(self):
        fdir = join(self.datadir,'nc')
        outdir = join(self.datadir,'tif')
        params = []
        for f in T.listdir(fdir):
            scale = f.split('.')[0]
            outdir_i = join(outdir,scale)
            T.mk_dir(outdir_i,force=True)
            fpath = join(fdir,f)
            param = [fpath,'spei',outdir_i]
            # self.kernel_nc_to_tif(param)
            # exit()
            params.append(param)
        MULTIPROCESS(self.kernel_nc_to_tif,params).run(process=7)

    def kernel_nc_to_tif(self,param):
        fpath, var, outdir_i = param
        self.nc_to_tif_func(fpath, var, outdir_i)
        pass


    def per_pix(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'per_pix')
        T.mk_dir(outdir)
        for folder in T.listdir(fdir):
            print(folder)
            fdir_i = join(fdir,folder)
            outdir_i = join(outdir,global_year_range,folder)
            T.mk_dir(outdir_i,force=True)
            Pre_Process().data_transform(fdir_i, outdir_i)

    def clean(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'per_pix_clean',global_year_range)
        T.mk_dir(outdir,force=True)

        for scale in T.listdir(fdir):
            outf = join(outdir,scale)
            fdir_i = join(fdir,scale)
            spatial_dict = T.load_npy_dir(fdir_i)
            spatial_dict_out = {}
            for pix in tqdm(spatial_dict,desc=scale):
                r,c = pix
                if r > 180:
                    continue
                vals = spatial_dict[pix]
                vals = np.array(vals)
                vals[vals<-999] = np.nan
                vals[vals>999] = np.nan
                if T.is_all_nan(vals):
                    continue
                spatial_dict_out[pix] = vals
            T.save_npy(spatial_dict_out, outf)

    def every_month(self):
        fdir = join(self.datadir,'per_pix_clean',global_year_range)
        outdir = join(self.datadir,'every_month',global_year_range)
        params_list = []
        for f in T.listdir(fdir):
            scale = f.split('.')[0]
            outdir_i = join(outdir, scale)
            T.mkdir(outdir_i, force=True)
            param = [fdir,f,outdir_i]
            # self.kernel_every_month(param)
            params_list.append(param)
        MULTIPROCESS(self.kernel_every_month,params_list).run(process=7)

    def kernel_every_month(self,params):
        fdir,f,outdir_i = params
        fpath = join(fdir, f)
        spatial_dict = T.load_npy(fpath)
        month_list = range(1, 13)
        for mon in month_list:
            spatial_dict_mon = {}
            for pix in tqdm(spatial_dict, desc=f'{mon}'):
                r, c = pix
                if r > 180:
                    continue
                vals = spatial_dict[pix]
                val_mon = T.monthly_vals_to_annual_val(vals, [mon])
                val_mon[val_mon < -10] = -999999
                num = T.count_num(val_mon, -999999)
                if num > 10:
                    continue
                val_mon[val_mon < -10] = np.nan
                if T.is_all_nan(val_mon):
                    continue
                spatial_dict_mon[pix] = val_mon
            outf = join(outdir_i, f'{mon:02d}')
            T.save_npy(spatial_dict_mon, outf)
        pass

    def nc_to_tif_func(self, fname, var_name, outdir):
        try:
            ncin = Dataset(fname, 'r')
            print(ncin.variables.keys())

        except:
            raise UserWarning('File not supported: ' + fname)
        try:
            lat = ncin.variables['lat'][:]
            lon = ncin.variables['lon'][:]
        except:
            try:
                lat = ncin.variables['latitude'][:]
                lon = ncin.variables['longitude'][:]
            except:
                try:
                    lat = ncin.variables['lat_FULL'][:]
                    lon = ncin.variables['lon_FULL'][:]
                except:
                    raise UserWarning('lat or lon not found')
        shape = np.shape(lat)
        try:
            time = ncin.variables['time_counter'][:]
            basetime_str = ncin.variables['time_counter'].units
        except:
            time = ncin.variables['time'][:]
            basetime_str = ncin.variables['time'].units

        basetime_unit = basetime_str.split('since')[0]
        basetime_unit = basetime_unit.strip()
        print(basetime_unit)
        print(basetime_str)
        if basetime_unit == 'days':
            timedelta_unit = 'days'
        elif basetime_unit == 'years':
            timedelta_unit = 'years'
        elif basetime_unit == 'month':
            timedelta_unit = 'month'
        elif basetime_unit == 'months':
            timedelta_unit = 'month'
        elif basetime_unit == 'seconds':
            timedelta_unit = 'seconds'
        elif basetime_unit == 'hours':
            timedelta_unit = 'hours'
        else:
            raise Exception('basetime unit not supported')
        basetime = basetime_str.strip(f'{timedelta_unit} since ')
        try:
            basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d')
        except:
            try:
                basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S')
            except:
                try:
                    basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S.%f')
                except:
                    try:
                        basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M')
                    except:
                        try:
                            basetime = datetime.datetime.strptime(basetime, '%Y-%m')
                        except:
                            raise UserWarning('basetime format not supported')
        data = ncin.variables[var_name]
        if len(shape) == 2:
            xx, yy = lon, lat
        else:
            xx, yy = np.meshgrid(lon, lat)
        for time_i in tqdm(range(len(time))):
            if basetime_unit == 'days':
                date = basetime + datetime.timedelta(days=int(time[time_i]))
            elif basetime_unit == 'years':
                date1 = basetime.strftime('%Y-%m-%d')
                base_year = basetime.year
                date2 = f'{int(base_year + time[time_i])}-01-01'
                delta_days = Tools().count_days_of_two_dates(date1, date2)
                date = basetime + datetime.timedelta(days=delta_days)
            elif basetime_unit == 'month' or basetime_unit == 'months':
                date1 = basetime.strftime('%Y-%m-%d')
                base_year = basetime.year
                base_month = basetime.month
                date2 = f'{int(base_year + time[time_i] // 12)}-{int(base_month + time[time_i] % 12)}-01'
                delta_days = Tools().count_days_of_two_dates(date1, date2)
                date = basetime + datetime.timedelta(days=delta_days)
            elif basetime_unit == 'seconds':
                date = basetime + datetime.timedelta(seconds=int(time[time_i]))
            elif basetime_unit == 'hours':
                date = basetime + datetime.timedelta(hours=int(time[time_i]))
            else:
                raise Exception('basetime unit not supported')
            time_str = time[time_i]
            mon = date.month
            year = date.year
            if year < 1982:
                continue
            # print(year)
            # exit()
            day = date.day
            outf_name = f'{year}{mon:02d}{day:02d}.tif'
            outpath = join(outdir, outf_name)
            if isfile(outpath):
                continue
            arr = data[time_i]
            arr = np.array(arr)
            lon_list = xx.flatten()
            lat_list = yy.flatten()
            val_list = arr.flatten()
            lon_list[lon_list > 180] = lon_list[lon_list > 180] - 360
            df = pd.DataFrame()
            df['lon'] = lon_list
            df['lat'] = lat_list
            df['val'] = val_list
            lon_list_new = df['lon'].tolist()
            lat_list_new = df['lat'].tolist()
            val_list_new = df['val'].tolist()
            DIC_and_TIF().lon_lat_val_to_tif(lon_list_new, lat_list_new, val_list_new, outpath)

    def pick_year_range(self):

        fdir = join(self.datadir,'per_pix_clean','1982-2015')
        year_range_list = []
        for VI in global_VIs_year_range_dict:
            year_range = global_VIs_year_range_dict[VI]
            if year_range == '1982-2015':
                continue
            outdir = join(self.datadir,'per_pix_clean',year_range)
            T.mk_dir(outdir)
            start_year = int(year_range.split('-')[0])
            end_year = int(year_range.split('-')[1])
            date_list = []
            for y in range(1982,2015 + 1):
                for m in range(1,13):
                    date = f'{y}-{m:02d}'
                    date_list.append(date)
            pick_date_list = []
            for y in range(start_year, end_year + 1):
                for m in range(1, 13):
                    date = f'{y}-{m:02d}'
                    pick_date_list.append(date)
            for f in T.listdir(fdir):
                fpath = join(fdir,f)
                outf = join(outdir,f)
                dic = T.load_npy(fpath)
                picked_vals_dic = {}
                for pix in tqdm(dic):
                    vals = dic[pix]
                    dic_i = dict(zip(date_list,vals))
                    picked_vals = []
                    for date in pick_date_list:
                        val = dic_i[date]
                        picked_vals.append(val)
                    picked_vals = np.array(picked_vals)
                    picked_vals_dic[pix] = picked_vals
                T.save_npy(picked_vals_dic,outf)


class SPI:
    def __init__(self):
        self.datadir = join(data_root,'SPI')
        pass

    def run(self):
        # self.cal_spi()
        self.pick_SPI_year_range()
        # self.every_month()
        # self.check_spi()
        pass

    def cal_spi(self):
        date_range = '1930-2020'
        data_start_year = 1930
        # P_dir = CRU().data_dir + 'pre/per_pix/'
        P_dir = join(Precipitation().datadir,'per_pix',date_range)
        # P_dic = T.load_npy_dir(P_dir,condition='005')
        P_dic = T.load_npy_dir(P_dir)
        # scale_list = [1,3,6,9,12]
        scale_list = range(1,25)
        for scale in scale_list:
            outdir = join(self.datadir,'per_pix',date_range)
            T.mk_dir(outdir,force=True)
            outf = join(outdir,f'spi{scale:02d}')
            # distrib = indices.Distribution('pearson')
            distrib = indices.Distribution('gamma')
            Periodicity = compute.Periodicity(12)
            spatial_dic = {}
            for pix in tqdm(P_dic,desc=f'scale {scale}'):
                r,c = pix
                if r > 180:
                    continue
                vals = P_dic[pix]
                vals = np.array(vals)
                vals = T.mask_999999_arr(vals,warning=False)
                if np.isnan(np.nanmean(vals)):
                    continue
                # zscore = Pre_Process().z_score_climatology(vals)
                spi = climate_indices.indices.spi(
                values=vals,
                scale=scale,
                distribution=distrib,
                data_start_year=data_start_year,
                calibration_year_initial=1960,
                calibration_year_final=2000,
                periodicity=Periodicity,
                # fitting_params: Dict = None,
                )
                spatial_dic[pix] = spi
                # plt.plot(spi)
                # plt.show()
            T.save_npy(spatial_dic,outf)

    def pick_SPI_year_range(self):
        fdir = join(self.datadir,'per_pix','1930-2020')
        # year_range = global_VIs_year_range_dict['NDVI3g']
        year_range = global_VIs_year_range_dict['CSIF']
        outdir = join(self.datadir,'per_pix',year_range)
        T.mk_dir(outdir)
        start_year = 1930
        end_year = 2020
        date_list = []
        for y in range(start_year,end_year + 1):
            for m in range(1,13):
                date = f'{y}-{m:02d}'
                date_list.append(date)
        pick_date_list = []
        pick_year_start = int(year_range.split('-')[0])
        pick_year_end = int(year_range.split('-')[1])
        for y in range(pick_year_start, pick_year_end + 1):
            for m in range(1, 13):
                date = f'{y}-{m:02d}'
                pick_date_list.append(date)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            dic = T.load_npy(fpath)
            picked_vals_dic = {}
            for pix in tqdm(dic,desc=f):
                vals = dic[pix]
                dic_i = dict(zip(date_list,vals))
                picked_vals = []
                for date in pick_date_list:
                    val = dic_i[date]
                    picked_vals.append(val)
                picked_vals = np.array(picked_vals)
                picked_vals_dic[pix] = picked_vals
            T.save_npy(picked_vals_dic,outf)

    def every_month(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'every_month',global_year_range)
        params_list = []
        for f in T.listdir(fdir):
            scale = f.split('.')[0]
            outdir_i = join(outdir, scale)
            T.mkdir(outdir_i, force=True)
            param = [fdir,f,outdir_i]
            # self.kernel_every_month(param)
            params_list.append(param)
        MULTIPROCESS(self.kernel_every_month,params_list).run(process=7)

    def kernel_every_month(self,params):
        fdir,f,outdir_i = params
        fpath = join(fdir, f)
        spatial_dict = T.load_npy(fpath)
        month_list = range(1, 13)
        for mon in month_list:
            spatial_dict_mon = {}
            for pix in spatial_dict:
                r, c = pix
                if r > 180:
                    continue
                vals = spatial_dict[pix]
                val_mon = T.monthly_vals_to_annual_val(vals, [mon])
                val_mon[val_mon < -10] = -999999
                num = T.count_num(val_mon, -999999)
                if num > 10:
                    continue
                val_mon[val_mon < -10] = np.nan
                if T.is_all_nan(val_mon):
                    continue
                spatial_dict_mon[pix] = val_mon
            outf = join(outdir_i, f'{mon:02d}')
            T.save_npy(spatial_dict_mon, outf)
        pass

    def check_spi(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            spatial_dict1 = {}
            for pix in spatial_dict:
                vals = spatial_dict[pix]
                spatial_dict1[pix] = len(vals)
                # spatial_dict1[pix] = np.mean(vals)
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict1)
            plt.imshow(arr)
            plt.show()


class Precipitation:

    def __init__(self):
        self.datadir = join(data_root,'CRU_precip')
        pass

    def run(self):
        # self.pick_year_range()
        # self.anomaly()
        # self.detrend()
        self.mean_annual_precipitation()
        # self.check_per_pix()
        pass

    def pick_year_range(self):
        fdir = join(self.datadir,'per_pix','1930-2020')
        outdir = join(self.datadir,'per_pix',global_year_range)
        T.mk_dir(outdir)
        outf = join(outdir,'precip')
        start_year = 1930
        end_year = 2020
        date_list = []
        for y in range(start_year, end_year + 1):
            for m in range(1, 13):
                date = f'{y}-{m:02d}'
                date_list.append(date)

        pick_date_list = []
        for y in range(global_start_year, global_end_year + 1):
            for m in range(1, 13):
                date = f'{y}-{m:02d}'
                pick_date_list.append(date)

        dic = T.load_npy_dir(fdir)
        picked_vals_dic = {}
        for pix in tqdm(dic):
            vals = dic[pix]
            vals = np.array(vals,dtype=np.float32)
            vals[vals < 0] = np.nan
            if T.is_all_nan(vals):
                continue
            dic_i = dict(zip(date_list, vals))
            picked_vals = []
            for date in pick_date_list:
                val = dic_i[date]
                picked_vals.append(val)
            picked_vals = np.array(picked_vals)
            picked_vals_dic[pix] = picked_vals
        T.save_npy(picked_vals_dic, outf)

    def anomaly(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'anomaly',global_year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)

    def detrend(self):
        fdir = join(self.datadir,'anomaly',global_year_range)
        outdir = join(self.datadir,'anomaly_detrend',global_year_range)
        T.mk_dir(outdir,force=True)
        outf = join(outdir,'precip.npy')
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_detrend = T.detrend_dic(spatial_dict)
        T.save_npy(spatial_dict_detrend,outf)

    def mean_annual_precipitation1(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        ourdir = join(self.datadir,'map')
        T.mk_dir(ourdir,force=True)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_mean = {}
        for pix in spatial_dict:
            vals = spatial_dict[pix]
            vals_mean = np.nanmean(vals) * 12
            spatial_dict_mean[pix] = vals_mean
            # plt.plot(vals)
            # plt.show()
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_mean)
        plt.imshow(arr,cmap='jet_r',vmin=0,vmax=2000,interpolation='nearest')
        plt.colorbar()
        plt.show()

        pass
    def mean_annual_precipitation(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        ourdir = join(self.datadir,'map')
        outf = join(ourdir,'map.tif')
        T.mk_dir(ourdir,force=True)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_mean = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            vals_reshape = np.array(vals).reshape(-1,12)
            vals_sum_all = []
            for vals_one_year in vals_reshape:
                vals_sum = np.nansum(vals_one_year)
                vals_sum_all.append(vals_sum)
            vals_mean = np.nanmean(vals_sum_all)
            spatial_dict_mean[pix] = vals_mean
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_mean)
        DIC_and_TIF().arr_to_tif(arr,outf)

        pass

    def check_per_pix(self):
        # fdir = join(self.datadir, 'per_pix', year_range)
        fdir = join(self.datadir, 'anomaly', global_year_range)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict1 = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            # a,b,r,p = T.nan_line_fit(list(range(len(vals))),vals)
            vals = np.array(vals,dtype=np.float)
            if type(vals) == float:
                continue
            vals[vals<-999] = np.nan
            if T.is_all_nan(vals):
                continue
            # spatial_dict1[pix] = np.mean(vals)
            spatial_dict1[pix] = len(vals)
            # spatial_dict1[pix] = a
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict1)
        plt.imshow(arr)
        plt.show()
        pass

class TMP:
    def __init__(self):
        self.datadir = join(data_root,'CRU_tmp')
        pass

    def run(self):
        # self.check_per_pix()
        # self.per_pix()
        # self.per_pix_2007_2020()
        # self.detrend()
        # self.anomaly()
        # self.anomaly_detrend()
        # self.anomaly_juping()
        # self.anomaly_juping_detrend()
        # self.mean_annual_temperature()
        self.max_annual_temperature()
        pass

    def per_pix(self):
        fdir = join(self.datadir,'tif',global_year_range)
        outdir = join(self.datadir,'per_pix',global_year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir,outdir)

    def per_pix_2007_2020(self):
        year_range = '2007-2020'
        fdir = join(self.datadir,'tif',global_year_range)
        outdir = join(self.datadir,'per_pix',year_range)
        start = 2007
        end = 2020
        T.mk_dir(outdir,force=True)
        picked_flist = []
        for f in T.listdir(fdir):
            year = f[:4]
            year = int(year)
            if year < start or year > end:
                continue
            picked_flist.append(f)
        Pre_Process().data_transform_with_date_list(fdir,outdir,picked_flist)

    def detrend(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'per_pix_detrend',global_year_range)
        T.mk_dir(outdir,force=True)
        outf = join(outdir,'detrend.npy')
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_detrend = T.detrend_dic(spatial_dict)
        T.save_npy(spatial_dict_detrend,outf)

    def anomaly(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'anomaly',global_year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)
        pass

    def anomaly_juping(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'anomaly_juping',global_year_range)
        T.mk_dir(outdir,force=True)

        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            spatial_dict_i = T.load_npy(fpath)
            result_dict_i = {}
            for pix in spatial_dict_i:
                vals = spatial_dict_i[pix]
                vals[vals<-999] = np.nan
                if T.is_all_nan(vals):
                    continue
                pix_anomaly = Pre_Process().climatology_anomaly(vals)
                result_dict_i[pix] = pix_anomaly
            outf = join(outdir,f)
            T.save_npy(result_dict_i,outf)
        pass

    def anomaly_juping_detrend(self):
        fdir = join(self.datadir,'anomaly_juping',global_year_range)
        outdir = join(self.datadir,'anomaly_juping_detrend',global_year_range)
        T.mk_dir(outdir,force=True)
        outf = join(outdir,'detrend.npy')
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_detrend = T.detrend_dic(spatial_dict)
        T.save_npy(spatial_dict_detrend,outf)

    def anomaly_detrend(self):
        fdir = join(self.datadir,'anomaly',global_year_range)
        outdir = join(self.datadir,'anomaly_detrend',global_year_range)
        T.mk_dir(outdir,force=True)
        outf = join(outdir,'detrend.npy')
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_detrend = T.detrend_dic(spatial_dict)
        T.save_npy(spatial_dict_detrend,outf)

    def check_per_pix(self):
        # fdir = join(self.datadir, 'per_pix', year_range)
        fdir = join(self.datadir, 'anomaly', global_year_range)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict1 = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            a,b,r,p = T.nan_line_fit(list(range(len(vals))),vals)
            # vals = np.array(vals)
            # vals[vals<-999] = np.nan
            # if T.is_all_nan(vals):
            #     continue
            # spatial_dict1[pix] = np.mean(vals)
            spatial_dict1[pix] = a
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict1)
        plt.imshow(arr)
        plt.show()
        pass


    def mean_annual_temperature(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'mat')
        T.mk_dir(outdir,force=True)
        outf = join(outdir,'mat_gs.tif')
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_mean = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            vals[vals<-999] = np.nan
            if T.is_all_nan(vals):
                continue
            vals_gs = T.monthly_vals_to_annual_val(vals,grow_season=global_gs)
            vals_mean = np.nanmean(vals_gs)
            spatial_dict_mean[pix] = vals_mean
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_mean)
        DIC_and_TIF().arr_to_tif(arr,outf)
        pass


    def max_annual_temperature(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'max_annual_temperature')
        T.mk_dir(outdir,force=True)
        # outf = join(outdir,'max_annual_temperature_gs.tif')
        outf = join(outdir,'max_annual_temperature.tif')
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_mean = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            vals[vals<-999] = np.nan
            if T.is_all_nan(vals):
                continue
            # vals_gs = T.monthly_vals_to_annual_val(vals,grow_season=global_gs,method='max')
            vals_gs = T.monthly_vals_to_annual_val(vals,grow_season=None,method='max')
            vals_mean = np.nanmean(vals_gs)
            spatial_dict_mean[pix] = vals_mean
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_mean)
        DIC_and_TIF().arr_to_tif(arr,outf)
        pass

class TMX:
    def __init__(self):
        self.datadir = join(data_root,'CRU_tmx')
        pass

    def run(self):
        # self.nc_to_tif()
        # self.check_per_pix()
        # self.per_pix_1982_2020()
        # self.per_pix_2007_2020()
        # self.detrend()
        # self.anomaly()
        # self.anomaly_detrend()
        # self.anomaly_juping()
        # self.anomaly_juping_detrend()
        # self.mean_annual_temperature()
        self.max_annual_temperature()
        pass

    def nc_to_tif(self):
        outdir = self.datadir + '/tif/'
        T.mk_dir(outdir,force=True)
        fdir = self.datadir + '/nc/'
        fpath = join(fdir,'cru_ts4.08.1901.2023.tmx.dat.nc')
        # print(year)
        # exit()
        ncin = Dataset(fpath, 'r')
        ncin_xarr = xr.open_dataset(fpath)
        # print(ncin.variables)
        # exit()
        lat = ncin['lat']
        lon = ncin['lon']
        pixelWidth = lon[1] - lon[0]
        pixelHeight = lat[1] - lat[0]
        pixelHeight = -pixelHeight
        longitude_start = -180
        latitude_start = 90
        time = ncin.variables['time']

        start = datetime.datetime(1900, 1, 1)
        # print(time)
        # for t in time:
        #     print(t)
        # exit()
        flag = 0
        for i in tqdm(range(len(time))):
            # print(i)
            flag += 1
            # print(time[i])
            date = start + datetime.timedelta(days=int(time[i]))
            if date.year < 1982:
                continue
            year = str(date.year)
            # exit()
            month = '%02d' % date.month
            day = '%02d'%date.day
            date_str = year + month
            newRasterfn = outdir + date_str + '.tif'
            if os.path.isfile(newRasterfn):
                continue
            # print(date_str)
            # exit()
            # if not date_str[:4] in valid_year:
            #     continue
            # print(date_str)
            # exit()
            # arr = ncin.variables['tmax'][i]
            arr = ncin_xarr['tmx'][i]
            arr = np.array(arr)[::-1]
            # print(arr)
            # grid = arr < 99999
            # arr[np.logical_not(grid)] = -999999
            newRasterfn = outdir + date_str + '.tif'
            ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
            # grid = np.ma.masked_where(grid>1000,grid)
            # DIC_and_TIF().arr_to_tif(arr,newRasterfn)
            # plt.imshow(arr,'RdBu')
            # plt.colorbar()
            # plt.show()
            # nc_dic[date_str] = arr
            # exit()

    def per_pix_1982_2020(self):
        year_range = '1982-2020'
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'per_pix',year_range)
        start = 1982
        end = 2020
        T.mk_dir(outdir,force=True)
        picked_flist = []
        for f in T.listdir(fdir):
            year = f[:4]
            year = int(year)
            if year < start or year > end:
                continue
            picked_flist.append(f)
        Pre_Process().data_transform_with_date_list(fdir,outdir,picked_flist)

    def per_pix_2007_2020(self):
        year_range = '2007-2020'
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'per_pix',year_range)
        start = 2007
        end = 2020
        T.mk_dir(outdir,force=True)
        picked_flist = []
        for f in T.listdir(fdir):
            year = f[:4]
            year = int(year)
            if year < start or year > end:
                continue
            picked_flist.append(f)
        Pre_Process().data_transform_with_date_list(fdir,outdir,picked_flist)

    def detrend(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'per_pix_detrend',global_year_range)
        T.mk_dir(outdir,force=True)
        outf = join(outdir,'detrend.npy')
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_detrend = T.detrend_dic(spatial_dict)
        T.save_npy(spatial_dict_detrend,outf)

    def anomaly(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'anomaly',global_year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)
        pass

    def anomaly_juping(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'anomaly_juping',global_year_range)
        T.mk_dir(outdir,force=True)

        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            spatial_dict_i = T.load_npy(fpath)
            result_dict_i = {}
            for pix in spatial_dict_i:
                vals = spatial_dict_i[pix]
                vals[vals<-999] = np.nan
                if T.is_all_nan(vals):
                    continue
                pix_anomaly = Pre_Process().climatology_anomaly(vals)
                result_dict_i[pix] = pix_anomaly
            outf = join(outdir,f)
            T.save_npy(result_dict_i,outf)
        pass

    def anomaly_juping_detrend(self):
        fdir = join(self.datadir,'anomaly_juping',global_year_range)
        outdir = join(self.datadir,'anomaly_juping_detrend',global_year_range)
        T.mk_dir(outdir,force=True)
        outf = join(outdir,'detrend.npy')
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_detrend = T.detrend_dic(spatial_dict)
        T.save_npy(spatial_dict_detrend,outf)

    def anomaly_detrend(self):
        fdir = join(self.datadir,'anomaly',global_year_range)
        outdir = join(self.datadir,'anomaly_detrend',global_year_range)
        T.mk_dir(outdir,force=True)
        outf = join(outdir,'detrend.npy')
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_detrend = T.detrend_dic(spatial_dict)
        T.save_npy(spatial_dict_detrend,outf)

    def check_per_pix(self):
        # fdir = join(self.datadir, 'per_pix', year_range)
        fdir = join(self.datadir, 'anomaly', global_year_range)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict1 = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            a,b,r,p = T.nan_line_fit(list(range(len(vals))),vals)
            # vals = np.array(vals)
            # vals[vals<-999] = np.nan
            # if T.is_all_nan(vals):
            #     continue
            # spatial_dict1[pix] = np.mean(vals)
            spatial_dict1[pix] = a
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict1)
        plt.imshow(arr)
        plt.show()
        pass


    def mean_annual_temperature(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'mat')
        T.mk_dir(outdir,force=True)
        outf = join(outdir,'mat_gs.tif')
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_mean = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            vals[vals<-999] = np.nan
            if T.is_all_nan(vals):
                continue
            vals_gs = T.monthly_vals_to_annual_val(vals,grow_season=global_gs)
            vals_mean = np.nanmean(vals_gs)
            spatial_dict_mean[pix] = vals_mean
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_mean)
        DIC_and_TIF().arr_to_tif(arr,outf)
        pass


    def max_annual_temperature(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'max_annual_temperature')
        T.mk_dir(outdir,force=True)
        # outf = join(outdir,'max_annual_temperature_gs.tif')
        outf = join(outdir,'max_annual_temperature.tif')
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_mean = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            vals[vals<-999] = np.nan
            if T.is_all_nan(vals):
                continue
            # vals_gs = T.monthly_vals_to_annual_val(vals,grow_season=global_gs,method='max')
            vals_gs = T.monthly_vals_to_annual_val(vals,grow_season=None,method='max')
            vals_mean = np.nanmean(vals_gs)
            spatial_dict_mean[pix] = vals_mean
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_mean)
        DIC_and_TIF().arr_to_tif(arr,outf)
        pass

class VPD:
    '''
    calculate from CRU
    '''
    def __init__(self):
        self.datadir = join(data_root, 'VPD')
        pass

    def run(self):
        # self.tif_to_perpix_1982_2020()
        # self.per_pix_clean()
        # self.anomaly()
        # self.detrend()
        self.check_per_pix()
        pass

    def __kernel_cal_VPD(self,tmn,tmx,vap):
        '''
        Worldwide impacts of atmospheric vapor pressure deficit on the interannual variability of terrestrial carbon sinks
        https://doi.org/10.1093/nsr/nwab150
        '''
        vap = vap * 0.1
        svp = 0.5 * (0.611 * np.exp((17.3*tmn)/(tmn+237.3)) + 0.611 * np.exp((17.3*tmx)/(tmx+237.3)))
        vpd = svp - vap

        return vpd

    def tif_to_perpix_1982_2020(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'perpix/1982-2020')
        T.mk_dir(outdir,force=True)
        selected_tif_list = []
        for y in range(1982,2021):
            for m in range(1,13):
                f = '{}{:02d}.tif'.format(y,m)
                selected_tif_list.append(f)
        Pre_Process().data_transform_with_date_list(fdir,outdir,selected_tif_list)

    def per_pix_clean(self):
        fdir = join(self.datadir,'perpix/1982-2020')
        outdir = join(self.datadir,'perpix_clean/1982-2020')
        T.mk_dir(outdir,force=True)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_clean = {}
        for pix in spatial_dict:
            vals = spatial_dict[pix]
            vals[vals<-999] = np.nan
            if T.is_all_nan(vals):
                continue
            spatial_dict_clean[pix] = vals
        outf = join(outdir,'VPD.npy')
        T.save_npy(spatial_dict_clean,outf)
        pass

    def anomaly(self):
        fdir = join(self.datadir,'perpix_clean',global_year_range)
        outdir = join(self.datadir,'anomaly',global_year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)
        pass

    def detrend(self):
        fdir = join(self.datadir,'anomaly',global_year_range)
        outdir = join(self.datadir,'detrend',global_year_range)
        T.mk_dir(outdir,force=True)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_detrend = T.detrend_dic(spatial_dict)
        outf = join(outdir,'VPD.npy')
        T.save_npy(spatial_dict_detrend,outf)

    def check_per_pix(self):
        fdir = join(self.datadir, 'per_pix',global_year_range)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict1 = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            # a,b,r,p = T.nan_line_fit(list(range(len(vals))),vals)
            vals = np.array(vals)
            vals[vals<0] = np.nan
            if T.is_all_nan(vals):
                continue
            plt.plot(vals)
            plt.show()
            # spatial_dict1[pix] = np.mean(vals)
            # spatial_dict1[pix] = a
            spatial_dict1[pix] = len(vals)
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict1)
        plt.imshow(arr)
        plt.show()
        pass

class GLC2000:

    def __init__(self):
        self.datadir = join(data_root,'GLC2000')
        pass

    def run(self):
        self.resample()
        # self.unify()
        self.reclass_lc()
        self.lc_dict_with_number()
        self.show_reclass_lc()
        self.show_lc_dict_with_number()
        pass

    def resample(self):

        tif = join(self.datadir,'glc2000_v1_1.tif')
        outtif = join(self.datadir,'glc2000_v1_1_05_deg.tif')
        ToRaster().resample_reproj(tif,outtif,res=0.5)

    def unify(self):
        tif = join(self.datadir,'glc2000_v1_1_05_deg.tif')
        outtif = join(self.datadir,'glc2000_v1_1_05_deg_unify.tif')
        DIC_and_TIF().unify_raster(tif,outtif)

    def reclass_lc(self):
        outf = join(self.datadir,'reclass_lc_dic2')
        excel = join(self.datadir,'glc2000_Global_Legend.xls')
        tif = join(self.datadir,'glc2000_v1_1_05_deg_unify.tif')
        legend_df = pd.read_excel(excel)
        val_dic = T.df_to_dic(legend_df,'VALUE')
        spatial_dic = DIC_and_TIF().spatial_tif_to_dic(tif)
        reclass_dic = {}
        for pix in spatial_dic:
            val = spatial_dic[pix]
            if np.isnan(val):
                continue
            val = int(val)
            # lc = val_dic[val]['reclass_1']
            lc = val_dic[val]['reclass_2']
            if type(lc) == float:
                continue
            reclass_dic[pix] = lc
        T.save_npy(reclass_dic,outf)

    def lc_dict_with_number(self):
        outf = join(self.datadir,'lc_dict_with_number.npy')
        tif = join(self.datadir,'glc2000_v1_1_05_deg_unify.tif')
        spatial_dic = DIC_and_TIF().spatial_tif_to_dic(tif)
        T.save_npy(spatial_dic,outf)

    def show_reclass_lc(self):
        lc_dict_f = join(self.datadir,'reclass_lc_dic.npy')
        lc_dict = T.load_npy(lc_dict_f)
        lc_list = []
        for pix in lc_dict:
            lc = lc_dict[pix]
            lc_list.append(lc)
        lc_list = list(set(lc_list))
        print(lc_list)

    def show_lc_dict_with_number(self):
        lc_dict_f = join(self.datadir,'lc_dict_with_number.npy')
        lc_dict = T.load_npy(lc_dict_f)
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(lc_dict)
        arr[np.isnan(arr)]=20
        dict_new = DIC_and_TIF().spatial_arr_to_dic(arr)
        T.save_npy(dict_new,lc_dict_f)


class CCI_SM:

    def __init__(self):
        self.datadir = join(data_root,'CCI-SM')
        pass

    def run(self):
        # self.per_pix()
        # self.per_pix_no_nan()
        # self.anomaly()
        self.detrend()
        # self.check_cci_sm()
        pass

    def per_pix(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'per_pix')
        T.mkdir(outdir)
        Pre_Process().data_transform(fdir,outdir)

    def per_pix_no_nan(self):
        fdir = join(self.datadir, 'per_pix')
        outdir = join(self.datadir, 'per_pix_no_nan')
        T.mk_dir(outdir)
        outf = join(outdir,'CCI-SM.npy')
        spatial_dic = T.load_npy_dir(fdir)
        spatial_dic1 = {}
        for pix in tqdm(spatial_dic):
            vals = spatial_dic[pix]
            vals = np.array(vals)
            vals[vals < -999] = np.nan
            if T.is_all_nan(vals):
                continue
            spatial_dic1[pix] = vals
        T.save_npy(spatial_dic1,outf)

    def anomaly(self):
        fdir = join(self.datadir,'per_pix_no_nan')
        outdir = join(self.datadir,'anomaly')
        T.mk_dir(outdir)
        Pre_Process().cal_anomaly(fdir,outdir)


    def detrend(self):
        f = join(self.datadir,'anomaly','CCI-SM.npy')
        outdir = join(self.datadir,'detrend')
        outf = join(outdir,'CCI-SM.npy')
        T.mk_dir(outdir)
        spatial_dict = T.load_npy(f)
        detrend_spatial_dict = T.detrend_dic(spatial_dict)
        T.save_npy(detrend_spatial_dict,outf)
        pass

    def check_cci_sm(self):
        fdir = join(self.datadir, 'anomaly')
        spatial_dic = T.load_npy_dir(fdir)
        spatial_dic1 = {}
        for pix in tqdm(spatial_dic):
            vals = spatial_dic[pix]
            vals = np.array(vals)
            vals[vals<-999] = np.nan
            if T.is_all_nan(vals):
                continue
            # mean = np.nanmean(vals)
            a,b,r,p = T.nan_line_fit(np.arange(len(vals)),vals)
            mean = len(vals)
            spatial_dic1[pix] = a
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic1)
        plt.imshow(arr)
        plt.show()

class CSIF:

    def __init__(self):
        self.datadir = join(data_root,'CSIF')
        self.year_range = global_VIs_year_range_dict['CSIF']
        pass

    def run(self):
        # self.nc_to_tif()
        # self.resample()
        # self.MVC()
        # self.per_pix()
        # self.anomaly()
        self.detrend()
        pass

    def nc_to_tif(self):
        fdir = join(self.datadir,'nc')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir)
        for year in T.listdir(fdir):
            folder = join(fdir,year)
            for f in tqdm(T.listdir(folder),desc=year):
                fpath = join(folder,f)
                date = f.split('.')[-3]
                outf = join(outdir,f'{date}.tif')
                ncin = Dataset(fpath, 'r')
                # print(ncin.variables.keys())
                arr = ncin['clear_daily_SIF'][::][::-1]
                # arr = ncin['clear_inst_SIF'][::][::-1]
                arr = np.array(arr)
                arr[arr<-10] = np.nan
                ToRaster().array2raster(outf, -180, 90, 0.05, -0.05, arr)
                # exit()

    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_05')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            ToRaster().resample_reproj(fpath,outf,0.5)

    def MVC(self):
        fdir = join(self.datadir,'tif_05')
        outdir = join(self.datadir,'tif_monthly')
        T.mk_dir(outdir)
        Pre_Process().monthly_compose(fdir,outdir,date_fmt='doy',method='max')

    def per_pix(self):
        fdir = join(self.datadir,'tif_monthly')
        outdir = join(self.datadir,'per_pix',self.year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir,outdir)

    def anomaly(self):
        fdir = join(self.datadir,'per_pix',self.year_range)
        outdir = join(self.datadir,'anomaly',self.year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)

    def detrend(self):
        fdir = join(self.datadir,'anomaly',self.year_range)
        outdir = join(self.datadir,'anomaly_detrend',self.year_range)
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            dic = T.load_npy(fpath)
            dic_detrend = T.detrend_dic(dic)
            T.save_npy(dic_detrend,outf)

class Terraclimate:
    def __init__(self):
        # self.datadir = join(data_root,'Terraclimate')
        self.datadir = this_root = '/home/liyang/Desktop/14T/yang/Aridity_index_calculate/'
        pass

    def run(self):
        self.nc_to_tif()
        # self.nc_to_tif_aet()
        # self.resample()
        # self.per_pix()
        # self.anomaly()
        # self.detrend()
        # self.download_all()
        pass

    def nc_to_tif(self):
        product = 'Precipitation'
        variable = 'ppt'
        outdir = self.datadir + f'data/{product}/tif/'
        # print(outdir);exit()
        T.mk_dir(outdir,force=True)
        fdir = self.datadir + f'data/{product}/nc/'
        for fi in T.listdir(fdir):
            print(fi)
            if fi.startswith('.'):
                continue
            f = fdir + fi
            year = fi.split('.')[-2].split('_')[-1]
            # print(year)
            # exit()
            ncin = Dataset(f, 'r')
            ncin_xarr = xr.open_dataset(f)
            # print(ncin.variables)
            # exit()
            lat = ncin['lat']
            lon = ncin['lon']
            pixelWidth = lon[1] - lon[0]
            pixelHeight = lat[1] - lat[0]
            longitude_start = lon[0]
            latitude_start = lat[0]
            time = ncin.variables['time']

            start = datetime.datetime(1900, 1, 1)
            # print(time)
            # for t in time:
            #     print(t)
            # exit()
            flag = 0
            for i in tqdm(range(len(time))):
                # print(i)
                flag += 1
                # print(time[i])
                date = start + datetime.timedelta(days=int(time[i]))
                year = str(date.year)
                # exit()
                month = '%02d' % date.month
                day = '%02d'%date.day
                date_str = year + month
                newRasterfn = outdir + date_str + '.tif'
                if os.path.isfile(newRasterfn):
                    continue
                # print(date_str)
                # exit()
                # if not date_str[:4] in valid_year:
                #     continue
                # print(date_str)
                # exit()
                # arr = ncin.variables['tmax'][i]
                arr = ncin_xarr[variable][i]
                arr = np.array(arr)
                # print(arr)
                # grid = arr < 99999
                # arr[np.logical_not(grid)] = -999999
                newRasterfn = outdir + date_str + '.tif'
                ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
                # grid = np.ma.masked_where(grid>1000,grid)
                # DIC_and_TIF().arr_to_tif(arr,newRasterfn)
                # plt.imshow(arr,'RdBu')
                # plt.colorbar()
                # plt.show()
                # nc_dic[date_str] = arr
                # exit()

    def nc_to_tif_aet(self):
        outdir = self.datadir + '/aet/tif/'
        T.mk_dir(outdir,force=True)
        fdir = self.datadir + '/aet/nc/'
        for fi in T.listdir(fdir):
            print(fi)
            if fi.startswith('.'):
                continue
            f = fdir + fi
            year = fi.split('.')[-2].split('_')[-1]
            # print(year)
            # exit()
            ncin = Dataset(f, 'r')
            ncin_xarr = xr.open_dataset(f)
            # print(ncin.variables)
            # exit()
            lat = ncin['lat']
            lon = ncin['lon']
            pixelWidth = lon[1] - lon[0]
            pixelHeight = lat[1] - lat[0]
            longitude_start = lon[0]
            latitude_start = lat[0]
            time = ncin.variables['time']

            start = datetime.datetime(1900, 1, 1)
            # print(time)
            # for t in time:
            #     print(t)
            # exit()
            flag = 0
            for i in tqdm(range(len(time))):
                # print(i)
                flag += 1
                # print(time[i])
                date = start + datetime.timedelta(days=int(time[i]))
                year = str(date.year)
                # exit()
                month = '%02d' % date.month
                day = '%02d'%date.day
                date_str = year + month
                newRasterfn = outdir + date_str + '.tif'
                if os.path.isfile(newRasterfn):
                    continue
                # print(date_str)
                # exit()
                # if not date_str[:4] in valid_year:
                #     continue
                # print(date_str)
                # exit()
                # arr = ncin.variables['tmax'][i]
                arr = ncin_xarr['aet'][i]
                arr = np.array(arr)
                # print(arr)
                # grid = arr < 99999
                # arr[np.logical_not(grid)] = -999999
                newRasterfn = outdir + date_str + '.tif'
                ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
                # grid = np.ma.masked_where(grid>1000,grid)
                # DIC_and_TIF().arr_to_tif(arr,newRasterfn)
                # plt.imshow(arr,'RdBu')
                # plt.colorbar()
                # plt.show()
                # nc_dic[date_str] = arr
                # exit()

    def resample(self):
        # var_i = 'aet'
        var_i = 'srad'
        fdir = join(self.datadir, f'{var_i}/tif')
        outdir = join(self.datadir, f'{var_i}/tif_05')
        T.mk_dir(outdir, force=True)
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            outpath = join(outdir, f)
            ToRaster().resample_reproj(fpath, outpath, res=0.5)
        pass

    def per_pix(self):
        # var_i = 'aet'
        var_i = 'srad'
        fdir = join(self.datadir, f'{var_i}/tif_05')
        outdir = join(self.datadir, f'{var_i}/per_pix')
        T.mk_dir(outdir, force=True)
        Pre_Process().data_transform(fdir, outdir)

    def anomaly(self):
        # var_i = 'aet'
        var_i = 'srad'
        fdir = join(self.datadir, f'{var_i}/per_pix')
        outdir = join(self.datadir, f'{var_i}/anomaly')
        T.mk_dir(outdir, force=True)
        Pre_Process().cal_anomaly(fdir, outdir)

    def detrend(self):
        # var_i = 'aet'
        var_i = 'srad'
        fdir = join(self.datadir,f'{var_i}/anomaly')
        outdir = join(self.datadir,f'{var_i}/anomaly_detrend')
        outf = join(outdir,f'{var_i}.npy')
        T.mk_dir(outdir)
        spatial_dict = T.load_npy_dir(fdir)
        detrend_spatial_dict = T.detrend_dic(spatial_dict)
        T.save_npy(detrend_spatial_dict,outf)
        pass

    def download_all(self):
        param_list = []
        # product_list = ['def','ws','vap','pdsi','pet','ppt','soil','tmax','vpd']
        # product_list = ['aet']
        # product_list = ['vpd']
        product_list = ['srad']
        for product in product_list:
            for y in range(1982, 2021):
                param_list.append([product,str(y)])
                params = [product,str(y)]
                # self.download(params)
        MULTIPROCESS(self.download, param_list).run(process=8, process_or_thread='t')

    def download(self,params):
        product, y = params
        outdir = join(self.datadir,product,'nc')
        T.mk_dir(outdir, force=True)
        url = 'https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_{}_{}.nc'.format(product, y)
        print(url)
        while 1:
            try:
                outf = join(outdir, '{}_{}.nc'.format(product, y))
                if os.path.isfile(outf):
                    return None
                req = requests.request('GET', url)
                content = req.content
                fw = open(outf, 'wb')
                fw.write(content)
                return None
            except Exception as e:
                print(url, 'error sleep 5s')
                time.sleep(5)



class ERA_SM:

    def __init__(self):
        self.datadir = data_root + 'ERA-SM/'

    def run(self):
        # self.download_sm()
        # self.nc_to_tif()
        # self.resample()
        # self.clean()
        # self.tif_to_perpix_1982_2020()
        self.anomaly()
        # self.detrend()
        # self.check_cci_sm()
        pass

    def download_sm(self):
        from ecmwfapi import ECMWFDataServer
        server = ECMWFDataServer()
        outdir = join(self.datadir,'nc')
        outf = join(outdir,'ERA_SM.nc')
        date_list = []
        for y in range(1982, 2016):
            for m in range(1, 13):
                date = '{}{:02d}{:02d}'.format(y, m, 1)
                date_list.append(date)
        date_str = '/'.join(date_list)
        # print date_str
        # exit()
        server.retrieve({
            "class": "ei",
            "dataset": "interim",
            "date": date_str,
            "expver": "1",
            "grid": "0.25/0.25",
            "levtype": "sfc",
            "param": "39.128",
            "stream": "moda",
            "type": "an",
            "target": outf,
            "format": "netcdf",
        })

        pass

    def nc_to_tif(self):
        f = join(self.datadir, 'nc/ERA_025.nc')
        outdir = join(self.datadir, 'tif')
        T.mk_dir(outdir)
        T.nc_to_tif(f,'swvl1',outdir)

    def resample(self):
        fdir = join(self.datadir,'tif','sum_all_layers')
        outdir = join(self.datadir,'tif_05')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            ToRaster().resample_reproj(fpath,outpath,res=0.5)

    def clean(self):
        fdir = join(self.datadir,'tif_05')
        outdir = join(self.datadir,'tif_05_clean')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            if not f.endswith('.tif'):
                continue
            outpath = join(outdir,f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array[array<=0] = np.nan
            ToRaster().array2raster(outpath, originX, originY, pixelWidth, pixelHeight, array)

    def tif_to_perpix_1982_2020(self):
        fdir = join(self.datadir,'tif_05')
        outdir = join(self.datadir,'per_pix/',global_year_range)
        T.mk_dir(outdir,force=True)
        selected_tif_list = []
        for y in range(global_start_year,global_end_year+1):
            for m in range(1,13):
                f = '{}{:02d}01.tif'.format(y,m)
                selected_tif_list.append(f)
        Pre_Process().data_transform_with_date_list(fdir,outdir,selected_tif_list)

    def anomaly(self):
        fdir = join(self.datadir, 'per_pix',global_year_range)
        outdir = join(self.datadir, 'anomaly',global_year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)
        pass

    def detrend(self):
        fdir = join(self.datadir,'anomaly/1982-2015')
        outdir = join(self.datadir,'detrend/1982-2015')
        T.mk_dir(outdir,force=True)
        outf = join(outdir,'ERA-SM.npy')
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_detrend = T.detrend_dic(spatial_dict)
        T.save_npy(spatial_dict_detrend,outf)

    def check_cci_sm(self):
        # fdir = join(self.datadir, 'anomaly','1982-2015')
        fdir = join(self.datadir, 'detrend','1982-2015')
        spatial_dic = T.load_npy_dir(fdir)
        spatial_dic1 = {}
        for pix in tqdm(spatial_dic):
            vals = spatial_dic[pix]
            vals = np.array(vals)
            vals[vals<-999] = np.nan
            if T.is_all_nan(vals):
                continue
            mean = np.nanmean(vals)
            # a,b,r,p = T.nan_line_fit(np.arange(len(vals)),vals)
            # mean = len(vals)
            # spatial_dic1[pix] = a
            spatial_dic1[pix] = mean
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic1)
        plt.imshow(arr)
        plt.show()

class GLEAM:

    def __init__(self):
        # self.product = 'SMroot'
        self.product = 'Et'
        self.datadir = data_root + f'GLEAM/{self.product}/'
        T.mk_dir(self.datadir)
        pass


    def run(self):
        # self.nc_to_tif()
        # self.resample()
        # self.tif_to_perpix_1982_2020()
        self.anomaly()
        # self.detrend()
        # self.check()
        pass


    def nc_to_tif(self):
        f = join(self.datadir,f'nc/{self.product}_1980-2020_GLEAM_v3.5a_MO.nc')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir)
        ncin = Dataset(f, 'r')
        ncin_xarr = xr.open_dataset(f)
        # print(ncin.variables)
        # exit()
        lat = ncin['lat']
        lon = ncin['lon']
        pixelWidth = lon[1] - lon[0]
        pixelHeight = lat[1] - lat[0]
        longitude_start = lon[0]
        latitude_start = lat[0]
        time_obj = ncin.variables['time']
        start = datetime.datetime(1900, 1, 1)
        # print(time)
        # for t in time:
        #     print(t)
        # exit()
        flag = 0
        for i in tqdm(range(len(time_obj))):
            # print(i)
            flag += 1
            # print(time[i])
            date = start + datetime.timedelta(days=int(time_obj[i]))
            year = str(date.year)
            # exit()
            month = '%02d' % date.month
            day = '%02d' % date.day
            date_str = year + month
            newRasterfn = join(outdir,date_str + '.tif')
            if os.path.isfile(newRasterfn):
                continue
            # print(date_str)
            # exit()
            # if not date_str[:4] in valid_year:
            #     continue
            # print(date_str)
            # exit()
            # arr = ncin.variables['pet'][i]
            arr = ncin_xarr.variables[self.product][i]
            arr = np.array(arr)
            arr[arr<0] = np.nan
            arr = arr.T
            # plt.imshow(arr)
            # plt.show()
            # print(arr)
            # grid = arr < 99999
            # arr[np.logical_not(grid)] = -999999
            ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
            # grid = np.ma.masked_where(grid>1000,grid)
            # DIC_and_TIF().arr_to_tif(arr,newRasterfn)
            # plt.imshow(arr,'RdBu')
            # plt.colorbar()
            # plt.show()
            # nc_dic[date_str] = arr
            # exit()

        pass

    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_05')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            ToRaster().resample_reproj(fpath,outpath,res=0.5)
        pass

    def tif_to_perpix_1982_2020(self):
        fdir = join(self.datadir,'tif_05')
        outdir = join(self.datadir,'per_pix/',global_year_range)
        T.mk_dir(outdir,force=True)
        selected_tif_list = []
        for y in range(global_start_year,global_end_year+1):
            for m in range(1,13):
                f = '{}{:02d}.tif'.format(y,m)
                selected_tif_list.append(f)
        Pre_Process().data_transform_with_date_list(fdir,outdir,selected_tif_list)

    def anomaly(self):
        fdir = join(self.datadir, 'per_pix',global_year_range)
        outdir = join(self.datadir, 'anomaly',global_year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)
        pass

    def detrend(self):
        fdir = join(self.datadir, 'anomaly',global_year_range)
        outdir = join(self.datadir, 'anomaly_detrend',global_year_range)
        T.mk_dir(outdir,force=True)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_detrend = T.detrend_dic(spatial_dict)
        outf = join(outdir,'detrend.npy')
        T.save_npy(spatial_dict_detrend,outf)
        pass

    def check(self):

        fdir = join(self.datadir,'per_pix',global_year_range)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_mean = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            mean = np.nanmean(vals)
            spatial_dict_mean[pix] = mean
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_mean)
        plt.imshow(arr,vmin=0,vmax=0.5,cmap='jet',interpolation='nearest')
        plt.colorbar()
        plt.show()

class ERA_2m_T:

    def __init__(self):
        self.datadir = join(data_root,'ERA_2m_T')
        pass

    def run(self):
        # self.download_data()
        # self.download_monthly()
        # self.nc_to_tif()
        # self.resample()
        # self.perpix()
        self.anomaly()
        pass

    def download_data(self):
        from ecmwfapi import ECMWFDataServer
        server = ECMWFDataServer()
        outdir = join(self.datadir,'nc')
        T.mk_dir(outdir,force=True)
        date_list = []
        init_date = datetime.datetime(1982,1,1)
        flag = 1
        for y in range(1982, 2019):
            for m in range(1, 13):
                # date = '{}{:02d}{:02d}'.format(y, m, 1)
                # date = "1982-01-01/to/1982-01-31"
                start_date_obj = T.month_index_to_date_obj(flag-1, init_date) - datetime.timedelta(days=1)
                end_date_obj = T.month_index_to_date_obj(flag, init_date) - datetime.timedelta(days=1)
                flag += 1
                start_date = start_date_obj.strftime('%Y-%m-%d')
                end_date = end_date_obj.strftime('%Y-%m-%d')
                date_range = f'{start_date}/to/{end_date}'
                # outf = join(outdir, f'{y}{m:02d}.nc')
                # print(date_range)
                date_list.append(date_range)
        for date_range in tqdm(date_list):
            # print(date_range)
            start_date = date_range.split('/')[0]
            end_date = date_range.split('/')[2]
            start_date = start_date.replace('-','')
            end_date = end_date.replace('-','')
            outf = join(outdir, f'{start_date}_{end_date}.nc')
            server.retrieve({
                "class": "ei",
                "dataset": "interim",
                "date": date_range,
                "expver": "1",
                "grid": "0.5/0.5",
                "levtype": "sfc",
                "param": "167.128",
                "step": "12",
                "stream": "oper",
                "time": "12:00:00",
                "type": "fc",
                "target": outf,
                "format": "netcdf",
            })
            # exit()
        pass

    def download_monthly(self):
        import cdsapi
        c = cdsapi.Client()
        outdir = join(self.datadir,'nc', 'monthly')
        T.mk_dir(outdir,force=True)
        date_list = []
        init_date = datetime.datetime(1982,1,1)
        flag = 1
        for y in range(1982, 2016):
            outf = join(outdir, f'{y}.nc')
            c.retrieve(
                'reanalysis-era5-single-levels-monthly-means',
                {
                    'format': 'netcdf',
                    'variable': '2m_temperature',
                    'year': f'{y}',
                    'month': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                    ],
                    'time': '00:00',
                    'product_type': 'monthly_averaged_reanalysis',
                },
                f'{outf}')
            # exit()
        pass

    def nc_to_tif(self):
        fdir = join(self.datadir,'nc','monthly')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir,force=True)

        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            T.nc_to_tif(fpath,'t2m',outdir)
        pass

    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_05')
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            ToRaster().resample_reproj(fpath,outf,0.5)

        pass

    def perpix(self):
        fdir = join(self.datadir,'tif_05')
        outdir = join(self.datadir,'perpix')
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir,outdir)

    def anomaly(self):
        fdir = join(self.datadir,'perpix')
        outdir = join(self.datadir,'anomaly')
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)

class ERA_Precip:

    def __init__(self):
        self.datadir = join(data_root,'ERA_Precip')
        pass

    def run(self):
        # self.download_monthly()
        # self.nc_to_tif()
        self.scale_offset()
        self.resample()
        self.perpix()
        self.anomaly()
        pass


    def download_monthly(self):
        import cdsapi
        c = cdsapi.Client()
        outdir = join(self.datadir,'nc', 'monthly')
        T.mk_dir(outdir,force=True)
        date_list = []
        init_date = datetime.datetime(1982,1,1)
        flag = 1
        for y in range(1982, 2016):
            outf = join(outdir, f'{y}.nc')
            c.retrieve(
                'reanalysis-era5-single-levels-monthly-means',
                {
                    'format': 'netcdf',
                    'variable': 'total_precipitation',
                    'year': f'{y}',
                    'month': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                    ],
                    'time': '00:00',
                    'product_type': 'monthly_averaged_reanalysis',
                },
                f'{outf}')
            # exit()
        pass
    def nc_to_tif(self):
        fdir = join(self.datadir,'nc','monthly')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir,force=True)

        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            T.nc_to_tif(fpath,'tp',outdir)
        pass

    def resample(self):
        fdir = join(self.datadir,'tif_offset')
        outdir = join(self.datadir,'tif_offset_05')
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            ToRaster().resample_reproj(fpath,outf,0.5)

        pass

    def scale_offset(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_offset')
        T.mk_dir(outdir,force=True)
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            date = f.split('.')[0]
            year,mon,day = date[:4],date[4:6],date[6:]
            year,mon,day = int(year),int(mon),int(day)
            days = T.number_of_days_in_month(year,mon)
            fpath = join(fdir,f)
            outf = join(outdir,f)
            arr = DIC_and_TIF().spatial_tif_to_arr(fpath)
            arr = arr * 1000 * days
            D = DIC_and_TIF(tif_template=fpath)
            D.arr_to_tif(arr,outf)
        pass

    def perpix(self):
        fdir = join(self.datadir,'tif_offset_05')
        outdir = join(self.datadir,'perpix','1982-2015')
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir,outdir)

    def anomaly(self):
        fdir = join(self.datadir,'perpix','1982-2015')
        outdir = join(self.datadir,'anomaly','1982-2015')
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)


class ERA_T2m_daily:

    def __init__(self):
        self.datadir = join(data_root,'ERA_daily/Tmax')
        # self.datadir = join(data_root,'ERA_daily/Tmean')

        pass

    def run(self):
        # self.unzip()
        # self.move_tif()
        # self.resample()
        # self.split_year()
        # self.per_pix_annual()
        self.daily_to_biweekly()
        # self.combine_annual_biweekly()
        # self.per_pix()
        # self.check_perpix()
        pass

    def unzip(self):
        fdir = join(self.datadir,'zips')
        outdir = join(self.datadir,'unzip')
        T.mk_dir(outdir,force=True)
        for folder in T.listdir(fdir):
            print(folder)
            folder_i = join(fdir,folder)
            T.unzip(folder_i,outdir)
        pass

    def move_tif(self):
        fdir = join(self.datadir,'unzip')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir,force=True)

        for folder in tqdm(T.listdir(fdir)):
            fpath = join(fdir,folder,f'download.temperature_2m.tif')
            outpath = join(outdir,f'{folder}.tif')
            shutil.move(fpath,outpath)


    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_05')
        T.mkdir(outdir)
        params_list = []
        for f in tqdm(T.listdir(fdir)):
            params = [fdir,outdir,f]
            params_list.append(params)
            # self.kernel_resample(params)
            # exit()
        MULTIPROCESS(self.kernel_resample,params_list).run(process=30)


    def kernel_resample(self,params):
        fdir,outdir,f = params
        fpath = join(fdir, f)
        outf = join(outdir, f)
        ToRaster().resample_reproj(fpath, outf, 0.5)
        pass


    def split_year(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_annual')
        T.mk_dir(outdir,force=True)

        for f in tqdm(T.listdir(fdir)):
            year = f[0:4]
            outdir_i = join(outdir,year)
            T.mk_dir(outdir_i,force=True)
            fpath = join(fdir,f)
            outf = join(outdir_i,f)
            shutil.move(fpath,outf)
        pass

    def per_pix_annual(self):
        fdir = join(self.datadir,'tif_annual')
        outdir = join(self.datadir,'perpix_annual')
        T.mk_dir(outdir,force=True)
        for year in T.listdir(fdir):
            print(year,'\n')
            fdir_i = join(fdir,year)
            outdir_i = join(outdir,year)
            T.mk_dir(outdir_i,force=True)
            Pre_Process().data_transform(fdir_i,outdir_i)
        pass

    def daily_to_biweekly(self):
        daily_temp_dir = join(self.datadir,'perpix_annual')
        outdir = join(self.datadir,'perpix_annual_biweekly_centos')
        T.mk_dir(outdir,force=True)
        params_list = []
        for year in T.listdir(daily_temp_dir):
            outdir_i = join(outdir,year)
            T.mk_dir(outdir_i,force=True)
            year_date_list = []
            base_date = datetime.datetime(int(year), 1, 1)
            for d in range(370):
                date_i = base_date + datetime.timedelta(days=d)
                year_date_i = date_i.year
                if not year_date_i == int(year):
                    break
                year_date_list.append(date_i)
            biweekly_date_index = []
            for mon in range(1,13):
                select_date_list = []
                for i,date_i in enumerate(year_date_list):
                    mon_i = date_i.month
                    if mon_i == mon:
                        select_date_list.append((i,date_i))
                part_a_index_list = []
                part_b_index_list = []
                for j,date_i in select_date_list:
                    day_i = date_i.day
                    if day_i <=15:
                        part_a_index_list.append(j)
                    else:
                        part_b_index_list.append(j)
                biweekly_date_index.append(part_a_index_list)
                biweekly_date_index.append(part_b_index_list)
            daily_temp_dir_i = join(daily_temp_dir,year)
            # for f in tqdm(T.listdir(daily_temp_dir_i),desc=year):
            for f in tqdm(T.listdir(daily_temp_dir_i),desc=year):
                params = [daily_temp_dir_i,f,biweekly_date_index,outdir_i]
                # self.kernel_daily_to_biweekly(params)
                params_list.append(params)
        MULTIPROCESS(self.kernel_daily_to_biweekly,params_list).run(process=22)

    def kernel_daily_to_biweekly(self,params):
        daily_temp_dir_i,f,biweekly_date_index,outdir_i = params
        fpath_temp = join(daily_temp_dir_i, f)
        temp_spatial_dict_i = T.load_npy(fpath_temp)
        spatial_dict = {}
        for pix in temp_spatial_dict_i:
            vals = temp_spatial_dict_i[pix]
            biweekly_vals = []
            for i in range(len(biweekly_date_index)):
                val_list = []
                for j in biweekly_date_index[i]:
                    val_list.append(vals[j])
                val = np.nanmean(val_list)
                biweekly_vals.append(val)
            spatial_dict[pix] = biweekly_vals
        outpath = join(outdir_i, f)
        T.save_npy(spatial_dict, outpath)

        pass

    def combine_annual_biweekly(self):
        fdir = join(self.datadir,'perpix_annual_biweekly')
        outdir = join(self.datadir,'perpix_biweekly')
        T.mk_dir(outdir,force=True)
        flist = []
        for year in T.listdir(fdir):
            folder = join(fdir,year)
            for f in T.listdir(folder):
                flist.append(f)
            break

        params_list = []
        for f in tqdm(flist):
            params = [fdir,f,outdir]
            params_list.append(params)

        MULTIPROCESS(self.kernel_combine_annual_biweekly,params_list).run(process=10)

        pass

    def kernel_combine_annual_biweekly(self,params):
        fdir,f,outdir = params
        spatial_dict = {}
        for year in T.listdir(fdir):
            fpath = join(fdir, year, f)
            spatial_dict_i = T.load_npy(fpath)
            for pix in spatial_dict_i:
                vals = spatial_dict_i[pix]
                if not pix in spatial_dict:
                    spatial_dict[pix] = []
                spatial_dict[pix].extend(vals)
        outf = join(outdir, f)
        T.save_npy(spatial_dict, outf)
        pass

    def check_perpix(self):
        fdir = join(self.datadir,'perpix_annual')
        for year in T.listdir(fdir):
            print(year)
            fdir_i = join(fdir,year)
            for f in T.listdir(fdir_i):
                fpath = join(fdir_i,f)
                arr = DIC_and_TIF().spatial_tif_to_arr(fpath)
                print(f,arr.shape)

        pass

class GPCC:

    def __init__(self):
        self.datadir = join(data_root, 'GPCC')
        pass

    def run(self):
        # self.download_monthly()
        # self.nc_to_tif()
        self.perpix()
        self.anomaly()
        pass

    def nc_to_tif(self):
        fdir = join(self.datadir,'nc')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir,force=True)

        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            T.nc_to_tif(fpath,'precip',outdir)
        pass

    def perpix(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'perpix')
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir,outdir)

    def anomaly(self):
        fdir = join(self.datadir,'perpix')
        outdir = join(self.datadir,'anomaly')
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)

class BEST:
    # Berkeley Earth Surface Temperatures (BEST)

    def __init__(self):
        self.datadir = join(data_root, 'Berkeley Earth Surface Temperatures')
        pass

    def run(self):
        # self.download_monthly()
        # self.nc_to_tif()
        # self.resample()
        self.perpix()
        # self.anomaly()
        pass

    def nc_to_tif(self):
        fdir = join(self.datadir,'nc')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir,force=True)

        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            self.__nc_to_tif(fpath,'temperature',outdir)
        pass

    def perpix(self):
        fdir = join(self.datadir,'tif_05')
        outdir = join(self.datadir,'perpix')
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir,outdir)

    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_05')
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            ToRaster().resample_reproj(fpath,outf,0.5)
        pass

    def anomaly(self):
        fdir = join(self.datadir,'perpix')
        outdir = join(self.datadir,'anomaly')
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)


    def __nc_to_tif(self, fname, var_name, outdir):
        try:
            ncin = Dataset(fname, 'r')
            print(ncin.variables.keys())

        except:
            raise UserWarning('File not supported: ' + fname)
        try:
            lat = ncin.variables['lat'][:]
            lon = ncin.variables['lon'][:]
        except:
            try:
                lat = ncin.variables['latitude'][:]
                lon = ncin.variables['longitude'][:]
            except:
                try:
                    lat = ncin.variables['lat_FULL'][:]
                    lon = ncin.variables['lon_FULL'][:]
                except:
                    raise UserWarning('lat or lon not found')
        shape = np.shape(lat)
        try:
            time = ncin.variables['time_counter'][:]
            basetime_str = ncin.variables['time_counter'].units
        except:
            time = ncin.variables['time'][:]
            basetime_str = ncin.variables['time'].units

        basetime_unit = 'month'
        print(basetime_unit)
        print(basetime_str)
        # basetime = basetime_str.strip(f'{timedelta_unit} since ')
        # basetime = '0000-00-00'
        basetime = datetime.datetime(1,1,1)
        data = ncin.variables[var_name]
        if len(shape) == 2:
            xx, yy = lon, lat
        else:
            xx, yy = np.meshgrid(lon, lat)
        for time_i in tqdm(range(len(time))):
            time_str = time[time_i]
            # print(time_str)
            # print(type(time_str))
            ratio = time_str - int(time_str)
            # print(ratio)
            mon = int(ratio * 12) + 1
            year = int(time_str)
            day = 1
            outf_name = f'{year}{mon:02d}{day:02d}.tif'
            outpath = join(outdir, outf_name)
            if isfile(outpath):
                continue
            arr = data[time_i]
            arr = np.array(arr)
            lon_list = xx.flatten()
            lat_list = yy.flatten()
            val_list = arr.flatten()
            lon_list[lon_list > 180] = lon_list[lon_list > 180] - 360
            df = pd.DataFrame()
            df['lon'] = lon_list
            df['lat'] = lat_list
            df['val'] = val_list
            lon_list_new = df['lon'].tolist()
            lat_list_new = df['lat'].tolist()
            val_list_new = df['val'].tolist()
            DIC_and_TIF().lon_lat_val_to_tif(lon_list_new, lat_list_new, val_list_new, outpath)

class GOME2_SIF:
    '''
    ref: Spatially downscaling sun-induced chlorophyll fluorescence leads to an improved temporal correlation with gross primary productivity
    '''
    def __init__(self):
        self.datadir = join(data_root, 'GOME2_SIF')
        pass

    def run(self):
        # self.nc_to_tif()
        # self.resample()
        # self.monthly_compose()
        # self.drop_invalid_value()
        # self.per_pix()
        # self.pick_year_range()
        # self.anomaly()
        self.detrend()
        pass

    def nc_to_tif(self):
        fdir = join(self.datadir,'nc')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outdir_i = join(outdir,f.split('.')[0])
            T.mk_dir(outdir_i,force=True)
            T.nc_to_tif(fpath,'SIF',outdir_i)

    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_05')
        T.mk_dir(outdir,force=True)
        for folder in T.listdir(fdir):
            fdir_i = join(fdir,folder)
            for f in tqdm(T.listdir(fdir_i),desc=folder):
                fpath = join(fdir_i,f)
                outpath = join(outdir,f)
                ToRaster().resample_reproj(fpath,outpath,0.5)
        pass

    def monthly_compose(self):
        fdir = join(self.datadir,'tif_05')
        outdir = join(self.datadir,'tif_mvc_05')
        T.mk_dir(outdir)
        Pre_Process().monthly_compose(fdir,outdir,method='max')
        pass

    def drop_invalid_value(self):
        '''
        nan value: -32768
        :return:
        '''
        fdir = join(self.datadir,'tif_mvc_05')
        outdir = join(self.datadir,'tif_mvc_05_drop_nan')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array[array<0] = np.nan
            DIC_and_TIF().arr_to_tif(array,outpath)

    def per_pix(self):
        fdir = join(self.datadir,'tif_mvc_05_drop_nan')
        outdir = join(self.datadir,'per_pix/2007-2018')
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir,outdir)

    def anomaly(self):
        # year_range = '2007-2018'
        year_range = '2007-2015'
        fdir = join(self.datadir,'per_pix',year_range)
        outdir = join(self.datadir,'anomaly',year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)

    def detrend(self):
        # year_range = '2007-2018'
        year_range = '2007-2015'
        fdir = join(self.datadir,'anomaly',year_range)
        outdir = join(self.datadir,'detrend',year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().detrend(fdir,outdir)

    def pick_year_range(self):
        origin_year_range = '2007-2018'
        year_range = '2007-2015'

        fdir = join(self.datadir,'per_pix',origin_year_range)
        year_range_list = []
        outdir = join(self.datadir,'per_pix',year_range)
        T.mk_dir(outdir)
        origin_start_year = int(origin_year_range.split('-')[0])
        origin_end_year = int(origin_year_range.split('-')[1])
        start_year = int(year_range.split('-')[0])
        end_year = int(year_range.split('-')[1])
        date_list = []
        for y in range(origin_start_year,origin_end_year + 1):
            for m in range(1,13):
                date = f'{y}-{m:02d}'
                date_list.append(date)
        pick_date_list = []
        for y in range(start_year, end_year + 1):
            for m in range(1, 13):
                date = f'{y}-{m:02d}'
                pick_date_list.append(date)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            dic = T.load_npy(fpath)
            picked_vals_dic = {}
            for pix in tqdm(dic):
                vals = dic[pix]
                dic_i = dict(zip(date_list,vals))
                picked_vals = []
                for date in pick_date_list:
                    val = dic_i[date]
                    picked_vals.append(val)
                picked_vals = np.array(picked_vals)
                picked_vals_dic[pix] = picked_vals
            T.save_npy(picked_vals_dic,outf)


class MODIS_LAI_Yuan:

    def __init__(self):
        self.data_dir = '/Users/liyang/Downloads/lai_monthly_yuanhua/'
        pass

    def run(self):
        # self.nc_to_tif()
        # self.per_pix()
        self.trend()
        pass

    def nc_to_tif(self):
        fdir = join(self.data_dir,'nc')
        outdir = join(self.data_dir,'tif')
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            year = f.split('.')[1].split('_')[1]
            fpath = join(fdir,f)
            ncin = Dataset(fpath, 'r')
            # print(ncin['time'])
            arrs = ncin['lai'][:]
            t = 1
            for arr in arrs:
                arr = np.array(arr,dtype=float)
                arr[arr<=0] = np.nan
                outf = join(outdir,f'{year}{t:02d}.tif')
                DIC_and_TIF().arr_to_tif(arr,outf)
                t += 1
        pass


    def per_pix(self):
        fdir = join(self.data_dir,'tif')
        outdir = join(self.data_dir,'per_pix')
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir,outdir)

    def trend(self):
        fdir = join(self.data_dir,'per_pix')
        outdir = join(self.data_dir,'trend')
        T.mk_dir(outdir,force=True)
        spatial_dict = T.load_npy_dir(fdir)
        mon_list = list(range(5,11))
        for m in tqdm(mon_list):
            trend_dict = {}
            gs = [m]
            for pix in tqdm(spatial_dict):
                vals = spatial_dict[pix]
                vals = T.mask_999999_arr(vals,warning=False)
                if T.is_all_nan(vals):
                    continue
                vals_gs = T.monthly_vals_to_annual_val(vals,gs)
                try:
                    a,b,r,p = T.nan_line_fit(np.arange(len(vals_gs)),vals_gs)
                    trend_dict[pix] = a
                except:
                    continue
            outf = join(outdir,f'{m}_trend.tif')
            DIC_and_TIF().pix_dic_to_tif(trend_dict,outf)


class MODIS_LAI_Chen:

    def __init__(self):
        self.data_dir = '/Volumes/NVME2T/greening_project_redo/data/MODIS_LAI/'
        pass

    def run(self):
        # self.monthly_compose()
        # self.perpix()
        self.trend()
        pass

    def monthly_compose(self):
        fdir = join(self.data_dir,'tif_05')
        outdir = join(self.data_dir,'monthly_compose')
        T.mk_dir(outdir,force=True)
        Pre_Process().monthly_compose(fdir,outdir,date_fmt='doy')

    def perpix(self):
        fdir = join(self.data_dir,'monthly_compose')
        outdir = join(self.data_dir,'per_pix_monthly')
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir,outdir)
        pass

    def trend(self):
        fdir = join(self.data_dir,'per_pix_monthly')
        outdir = join(self.data_dir,'trend')
        T.mk_dir(outdir,force=True)
        spatial_dict = T.load_npy_dir(fdir)
        mon_list = list(range(5,11))
        for m in tqdm(mon_list):
            trend_dict = {}
            gs = [m]
            for pix in tqdm(spatial_dict):
                r,c = pix
                if r > 180:
                    continue
                vals = spatial_dict[pix]
                vals = T.mask_999999_arr(vals,warning=False)
                if T.is_all_nan(vals):
                    continue
                # print(len(vals))
                # plt.plot(vals)
                # plt.show()
                vals_gs = T.monthly_vals_to_annual_val(vals,gs)
                try:
                    a,b,r,p = T.nan_line_fit(np.arange(len(vals_gs)),vals_gs)
                    trend_dict[pix] = a
                except:
                    continue
            outf = join(outdir,f'{m}_trend_chen.tif')
            DIC_and_TIF().pix_dic_to_tif(trend_dict,outf)
        pass


class CCI_SM_v7:
    def __init__(self):
        self.datadir = join(data_root, 'CCI_SM_v7')
        pass

    def run(self):
        # self.download()
        # self.check_download_nc()

        # self.nc_to_tif()
        # self.tif_clean()
        # self.monthly_compose()
        # self.resample()
        self.per_pix()
        # self.anomaly()
        # self.anomaly_detrend()
        pass

    def download(self):
        outdir = join(self.datadir,'nc')
        year_range_list = global_year_range_list
        params_list = []
        for year in year_range_list:
            params = [outdir,year]
            params_list.append(params)
            # self.kernel_download(params)
        MULTIPROCESS(self.kernel_download,params_list).run(process=10, process_or_thread='t')


    def kernel_download(self,params):
        outdir,year = params
        url = f'https://data.ceda.ac.uk/neodc/esacci/soil_moisture/data/daily_files/COMBINED/v07.1/{year}?json'
        req = requests.request('GET', url)
        content = req.content
        json_obj = json.loads(content)
        json_formatted_str = json.dumps(json_obj, indent=4)
        items = json_obj['items']
        for i in range(len(items)):
            dict_i = dict(items[i])
            download_url = dict_i['download']
            path = dict_i['path']
            name = dict_i['name']
            path2 = path.split('/')[-4:-1]
            outdir_i = join(outdir, *path2)
            outpath = join(outdir_i, name)
            T.mk_dir(outdir_i, force=True)
            self.download_f(download_url, outpath)

    def download_f(self,url,outf):
        if os.path.isfile(outf):
            return None
        req = requests.request('GET', url)
        content = req.content
        fw = open(outf, 'wb')
        fw.write(content)

    def check_download_nc(self):
        fdir = join(self.datadir,'nc','COMBINED/v07.1')
        for year in T.listdir(fdir):
            folder = join(fdir,year)
            for f in tqdm(T.listdir(folder),desc=year):
                fpath = join(folder,f)
                try:
                    nc = Dataset(fpath,'r')
                except:
                    os.remove(fpath)
                    print(fpath)

    def nc_to_tif(self):
        fdir = join(self.datadir,'nc','COMBINED/v07.1')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir,force=True)
        param_list = []
        for year in T.listdir(fdir):
            folder = join(fdir,year)
            for f in tqdm(T.listdir(folder),desc=year):
                fpath = join(folder,f)
                outdir_i = join(outdir,year)
                T.mk_dir(outdir_i,force=True)
                outf = join(outdir,outdir_i)
                param = [fpath,'sm',outdir_i]
                param_list.append(param)
                # self._nc_to_tif(param)
        MULTIPROCESS(self._nc_to_tif,param_list).run(process=6,process_or_thread='p')


    def _nc_to_tif(self, params):
        fname, var_name, outdir = params
        try:
            ncin = Dataset(fname, 'r')
            # print(ncin.variables.keys())

        except:
            raise UserWarning('File not supported: ' + fname)
        try:
            lat = ncin.variables['lat'][:]
            lon = ncin.variables['lon'][:]
        except:
            try:
                lat = ncin.variables['latitude'][:]
                lon = ncin.variables['longitude'][:]
            except:
                try:
                    lat = ncin.variables['lat_FULL'][:]
                    lon = ncin.variables['lon_FULL'][:]
                except:
                    raise UserWarning('lat or lon not found')
        shape = np.shape(lat)
        try:
            time = ncin.variables['time_counter'][:]
            basetime_str = ncin.variables['time_counter'].units
        except:
            time = ncin.variables['time'][:]
            basetime_str = ncin.variables['time'].units

        basetime_unit = basetime_str.split('since')[0]
        basetime_unit = basetime_unit.strip()
        # print(basetime_unit)
        # print(basetime_str)
        if basetime_unit == 'days':
            timedelta_unit = 'days'
        elif basetime_unit == 'years':
            timedelta_unit = 'years'
        elif basetime_unit == 'month':
            timedelta_unit = 'month'
        elif basetime_unit == 'months':
            timedelta_unit = 'month'
        elif basetime_unit == 'seconds':
            timedelta_unit = 'seconds'
        elif basetime_unit == 'hours':
            timedelta_unit = 'hours'
        else:
            raise Exception('basetime unit not supported')
        basetime = basetime_str.strip(f'{timedelta_unit} since ')
        basetime = basetime.split(' ')[0]
        # exit()
        try:
            basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d')
        except:
            try:
                basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S')
            except:
                try:
                    basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S.%f')
                except:
                    try:
                        basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M')
                    except:
                        try:
                            basetime = datetime.datetime.strptime(basetime, '%Y-%m')
                        except:
                            raise UserWarning('basetime format not supported')
        data = ncin.variables[var_name]
        if len(shape) == 2:
            xx, yy = lon, lat
        else:
            xx, yy = np.meshgrid(lon, lat)
        for time_i in range(len(time)):
            if basetime_unit == 'days':
                date = basetime + datetime.timedelta(days=int(time[time_i]))
            elif basetime_unit == 'years':
                date1 = basetime.strftime('%Y-%m-%d')
                base_year = basetime.year
                date2 = f'{int(base_year + time[time_i])}-01-01'
                delta_days = Tools().count_days_of_two_dates(date1, date2)
                date = basetime + datetime.timedelta(days=delta_days)
            elif basetime_unit == 'month' or basetime_unit == 'months':
                date1 = basetime.strftime('%Y-%m-%d')
                base_year = basetime.year
                base_month = basetime.month
                date2 = f'{int(base_year + time[time_i] // 12)}-{int(base_month + time[time_i] % 12)}-01'
                delta_days = Tools().count_days_of_two_dates(date1, date2)
                date = basetime + datetime.timedelta(days=delta_days)
            elif basetime_unit == 'seconds':
                date = basetime + datetime.timedelta(seconds=int(time[time_i]))
            elif basetime_unit == 'hours':
                date = basetime + datetime.timedelta(hours=int(time[time_i]))
            else:
                raise Exception('basetime unit not supported')
            time_str = time[time_i]
            mon = date.month
            year = date.year
            day = date.day
            outf_name = f'{year}{mon:02d}{day:02d}.tif'
            outpath = join(outdir, outf_name)
            if isfile(outpath):
                continue
            arr = data[time_i]
            arr = np.array(arr)
            lon_list = xx.flatten()
            lat_list = yy.flatten()
            val_list = arr.flatten()
            lon_list[lon_list > 180] = lon_list[lon_list > 180] - 360
            df = pd.DataFrame()
            df['lon'] = lon_list
            df['lat'] = lat_list
            df['val'] = val_list
            lon_list_new = df['lon'].tolist()
            lat_list_new = df['lat'].tolist()
            val_list_new = df['val'].tolist()
            DIC_and_TIF().lon_lat_val_to_tif(lon_list_new, lat_list_new, val_list_new, outpath)


    def tif_clean(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_clean')
        T.mk_dir(outdir,force=True)
        for year in T.listdir(fdir):
            folder = join(fdir,year)
            for f in tqdm(T.listdir(folder),desc=year):
                fpath = join(folder,f)
                outpath = join(outdir,f)
                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
                array[array<0] = np.nan
                ToRaster().array2raster(outpath, originX, originY, pixelWidth, pixelHeight, array)

    def monthly_compose(self):
        fdir = join(self.datadir,'tif_clean')
        outdir = join(self.datadir,'tif_monthly')
        T.mk_dir(outdir,force=True)
        Pre_Process().monthly_compose(fdir,outdir)

    def resample(self):
        fdir = join(self.datadir,'tif_monthly')
        outdir = join(self.datadir,'tif_monthly_05')
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            ToRaster().resample_reproj(fpath,outf,0.5)

    def per_pix(self):
        fdir = join(self.datadir,'tif_monthly_05')
        outdir = join(self.datadir,'per_pix',global_year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir,outdir)

    def anomaly(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'anomaly',global_year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)

    def anomaly_detrend(self):
        fdir = join(self.datadir,'anomaly',global_year_range)
        outdir = join(self.datadir,'anomaly_detrend',global_year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().detrend(fdir,outdir)

class FAPAR:

    def __init__(self):
        self.datadir = join(data_root, 'FAPAR')
        pass

    def run(self):
        # self.download()
        # self.check_nc()
        # self.nc_to_tif()
        # self.resample()
        # self.monthly_compose()
        # self.per_pix()
        # self.anomaly()
        self.detrend()
        pass

    def download(self):
        from bs4 import BeautifulSoup
        outdir = join(self.datadir,'nc')
        father_url = 'https://www.ncei.noaa.gov/data/avhrr-land-leaf-area-index-and-fapar/access/'
        year_list = global_year_range_list
        for year in year_list:
            outdir_i = join(outdir,f'{year}')
            T.mk_dir(outdir_i,force=True)
            url = father_url + f'{year}/'
            url_html = requests.get(url)
            soup = BeautifulSoup(url_html.text, 'html.parser')
            nc_list = []
            for link in soup.find_all('a'):
                link = link.get('href')
                if link.endswith('.nc'):
                    nc_list.append(link)
            param_list = []
            for i in range(len(nc_list)):
                url_i = url + nc_list[i]
                fpath_i = join(outdir_i,nc_list[i])
                param = [url_i,fpath_i]
                param_list.append(param)
                # self.kernel_download(param)
            print(year)
            MULTIPROCESS(self.kernel_download,param_list).run(process=10,process_or_thread='t')

    def kernel_download(self,param):
        url, outf = param
        while 1:
            try:
                if os.path.isfile(outf):
                    return None
                req = requests.request('GET', url)
                content = req.content
                fw = open(outf, 'wb')
                fw.write(content)
                return None
            except Exception as e:
                print(url, 'error sleep 5s')
                time.sleep(5)

    def check_nc(self):
        fdir = join(self.datadir,'nc')
        for year in T.listdir(fdir):
            folder = join(fdir,year)
            for f in tqdm(T.listdir(folder),desc=year):
                fpath = join(folder,f)
                try:
                    nc = Dataset(fpath,'r')
                except:
                    print(fpath)

    def nc_to_tif(self):
        fdir = join(self.datadir,'nc')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir,force=True)
        param_list = []
        for year in T.listdir(fdir):
            param = [fdir,year,outdir]
            self.kernel_nc_to_tif(param)
            # param_list.append(param)
        # MULTIPROCESS(self.kernel_nc_to_tif,param_list).run(process=1,process_or_thread='p')

    def kernel_nc_to_tif(self,param):
        fdir,year,outdir = param
        # outdir1 = '/Users/liyang/Projects_data/temp_dir'
        # outdir1_i = join(outdir1, year)
        # T.mk_dir(outdir1_i, force=True)
        folder = join(fdir, year)
        outdir_i = join(outdir, year)
        T.mk_dir(outdir_i, force=True)
        # for f in T.listdir(folder):
        for f in tqdm(T.listdir(folder),desc=year):
            date = f.split('_')[-2]
            outf = join(outdir_i, f'{date}.tif')
            # outf_1 = join(outdir1_i, f'{date}.tif')
            if isfile(outf):
                continue
            fpath = join(folder, f)
            ncin = Dataset(fpath, 'r')
            # print(ncin['time'])
            arrs = ncin['FAPAR'][:]
            for arr in arrs:
                arr = np.array(arr, dtype=float)
                arr[arr <= 0] = np.nan
                ToRaster().array2raster(outf, -180, 90, 0.05, -0.05, arr)
                ToRaster().resample_reproj(outf,outf,0.5)

    def resample(self):
        fdir = join(self.datadir,'tif')
        for year in T.listdir(fdir):
            for f in tqdm(T.listdir(join(fdir,year)),desc=year):
                fpath = join(fdir,year,f)
                outf = join(fdir,year,f)
                ToRaster().resample_reproj(fpath,outf,0.5)
                # exit()

    def monthly_compose(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_monthly')
        T.mk_dir(outdir,force=True)
        for year in T.listdir(fdir):
            folder = join(fdir,year)
            Pre_Process().monthly_compose(folder,outdir,date_fmt='yyyymmdd')

    def per_pix(self):
        fdir = join(self.datadir,'tif_monthly')
        outdir = join(self.datadir,'per_pix',global_year_range)
        T.mk_dir(outdir,force=True)
        selected_tif_list = []
        for y in global_year_range_list:
            for m in range(1, 13):
                f = '{}{:02d}.tif'.format(y, m)
                selected_tif_list.append(f)
        Pre_Process().data_transform_with_date_list(fdir, outdir, selected_tif_list)

    def anomaly(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'anomaly',global_year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)

    def detrend(self):
        fdir = join(self.datadir,'anomaly',global_year_range)
        outdir = join(self.datadir,'anomaly_detrend',global_year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().detrend(fdir,outdir)


class Aridity_Index:

    def __init__(self):
        # self.datadir = join(data_root, 'Aridity_Index')
        self.datadir = this_root = '/home/liyang/Desktop/14T/yang/Aridity_index_calculate/'
        pass

    def run(self):
        # self.Binary_tif()
        # self.plot_binary_tif()
        self.cal_Aridity_index()
        pass

    def cal_tif_dir_sum(self,product):
        fdir = join(self.datadir,'data',product,'tif')
        outdir = join(self.datadir,'data',product,'sum')
        T.mk_dir(outdir,force=True)
        outf = join(outdir,'sum.tif')
        if isfile(outf):
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(outf)
            return arr, originX, originY, pixelWidth, pixelHeight

        sum = 0.
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            sum += arr
        ToRaster().array2raster(outf, originX, originY, pixelWidth, pixelHeight, sum)
        return sum, originX, originY, pixelWidth, pixelHeight

        pass

    def cal_Aridity_index(self):
        # fdir_pet = join(self.datadir,'data','PET/tif')
        sum_pet, originX, originY, pixelWidth, pixelHeight = self.cal_tif_dir_sum('PET')
        sum_ppt, originX, originY, pixelWidth, pixelHeight = self.cal_tif_dir_sum('Precipitation')
        AI = sum_ppt / sum_pet
        AI[AI<0] = np.nan
        AI[AI>10] = np.nan
        outdir = join(self.datadir,'Aridity_index')
        T.mk_dir(outdir)
        ouf = join(outdir,'AI_4km.tif')
        ToRaster().array2raster(ouf, originX, originY, pixelWidth, pixelHeight, AI)
        # plt.imshow(AI,vmin=0,vmax=3,cmap='jet_r',interpolation='nearest')
        # plt.colorbar()
        # plt.show()


    def Binary_tif(self):
        fpath = join(self.datadir,'aridity_index.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
        binary_spatial_dict = {}
        for pix in tqdm(spatial_dict):
            val = spatial_dict[pix]
            if np.isnan(val):
                continue
            if val < 0.65:
                binary_spatial_dict[pix] = 0
            else:
                binary_spatial_dict[pix] = 1
        outf = join(self.datadir,'aridity_index_binary.tif')
        DIC_and_TIF().pix_dic_to_tif(binary_spatial_dict,outf)

    def plot_binary_tif(self):
        fpath = join(self.datadir,'aridity_index_binary.tif')
        Plot().plot_ortho(fpath,cmap='RdBu',vmin=-.3,vmax=1.3)
        outf = join(self.datadir,'aridity_index_binary.png')
        plt.savefig(outf,dpi=300)
        plt.close()
        T.open_path_and_file(self.datadir)
        pass


class TCSIF:

    def __init__(self):
        self.datadir = join(data_root, 'TCSIF')
        pass

    def run(self):
        # self.tif_clean()
        # self.rename()
        self.per_pix()
        pass

    def tif_clean(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_clean')
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            outf = join(outdir,f)
            arr = DIC_and_TIF().spatial_tif_to_arr(fpath)
            arr[arr<0] = np.nan
            outf = join(outdir,f)
            DIC_and_TIF().arr_to_tif(arr,outf)

    def rename(self):
        fdir = join(self.datadir,'tif_clean')
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            date = f.split('.')[0].split('_')[-1]
            outf = join(fdir,f'{date}.tif')
            os.rename(fpath,outf)

    def per_pix(self):
        fdir = join(self.datadir,'tif_clean')
        year_range = '2007-2020'
        start_year = int(year_range.split('-')[0])
        end_year = int(year_range.split('-')[1])
        outdir = join(self.datadir,'per_pix',year_range)
        T.mk_dir(outdir,force=True)
        selected_tif_list = []
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            year = f.split('.')[0][:4]
            year = int(year)
            if year < start_year or year > end_year:
                continue
            selected_tif_list.append(f)
        Pre_Process().data_transform_with_date_list(fdir, outdir, selected_tif_list)


class Global_Ecological_Zone:

    def __init__(self):
        self.datadir = join(data_root, 'Global_Ecological_Zone')
        pass

    def run(self):
        # self.rasterize()
        # self.legend()
        self.plot_ortho_GEZ_map()
        pass

    def rasterize(self):
        fpath = join(self.datadir,'gez2010/gez_2010_wgs84.shp')
        outdir = join(self.datadir,'tif')
        in_raster_template = join(this_root,'conf/land.tif')
        T.mk_dir(outdir)
        outf = join(outdir,'gez_2010.tif')
        self.shp_to_raster(fpath,outf,0.5,in_raster_template)

        pass

    def shp_to_raster(self, in_shp, output_raster, pixel_size, in_raster_template=None, ndv=-999999):
        input_shp = ogr.Open(in_shp)
        shp_layer = input_shp.GetLayer()
        if in_raster_template:
            raster = gdal.Open(in_raster_template)
            ulx, xres, xskew, uly, yskew, yres = raster.GetGeoTransform()
            lrx = ulx + (raster.RasterXSize * xres)
            lry = uly + (raster.RasterYSize * yres)
            xmin, xmax, ymin, ymax = ulx, lrx, lry, uly
        else:
            xmin, xmax, ymin, ymax = shp_layer.GetExtent()
        ds = gdal.Rasterize(output_raster, in_shp, xRes=pixel_size, yRes=pixel_size,
                            noData=ndv, outputBounds=[xmin, ymin, xmax, ymax],
                            outputType=gdal.GDT_Float32,attribute='gez_code')
        ds = None
        return output_raster

    def legend(self):
        fpath = join(self.datadir,'legend.xlsx')
        df = pd.read_excel(fpath)
        T.print_head_n(df)
        legend_dict = {}
        gez_name = df['gez_name'].tolist()
        gez_code = df['gez_code'].tolist()
        for i in range(len(gez_name)):
            legend_dict[gez_code[i]] = gez_name[i]
        outf = join(self.datadir,'legend')
        T.save_npy(legend_dict,outf)
        pass

    def plot_ortho_GEZ_map(self):
        tif = join(self.datadir,'tif/gez_2010.tif')
        arr = DIC_and_TIF().spatial_tif_to_arr(tif)
        plt.imshow(arr,cmap='Spectral')
        plt.colorbar()
        plt.show()
        pass

class IPCC_cliamte_zone:

    def __init__(self):
        self.datadir = join(data_root, 'IPCC_cliamte_zone')
        pass

    def run(self):
        self.rasterize()
        # self.legend()
        pass

    def rasterize(self):
        # fpath = join(self.datadir,r"IPCC-WGI-reference-regions-v4_shapefile\IPCC-WGI-reference-regions-v4_shapefile\IPCC-WGI-reference-regions-v4.shp")
        fpath = join(self.datadir,r"v3\shp\referenceRegions.shp")
        # print(fpath);exit()
        outdir = join(self.datadir,'tif')
        in_raster_template = join(this_root,'conf/land.tif')
        T.mk_dir(outdir)
        outf = join(outdir,'IPCC-WGI-v3.tif')
        self.shp_to_raster(fpath,outf,0.5,in_raster_template)

        pass

    def shp_to_raster(self, in_shp, output_raster, pixel_size, in_raster_template=None, ndv=-999999):
        input_shp = ogr.Open(in_shp)
        shp_layer = input_shp.GetLayer()
        if in_raster_template:
            raster = gdal.Open(in_raster_template)
            ulx, xres, xskew, uly, yskew, yres = raster.GetGeoTransform()
            lrx = ulx + (raster.RasterXSize * xres)
            lry = uly + (raster.RasterYSize * yres)
            xmin, xmax, ymin, ymax = ulx, lrx, lry, uly
        else:
            xmin, xmax, ymin, ymax = shp_layer.GetExtent()
        ds = gdal.Rasterize(output_raster, in_shp, xRes=pixel_size, yRes=pixel_size,
                            noData=ndv, outputBounds=[xmin, ymin, xmax, ymax],
                            outputType=gdal.GDT_Float32,attribute='ID')
        ds = None
        return output_raster

    def legend(self):
        fpath = join(self.datadir,'legend.xlsx')
        df = pd.read_excel(fpath)
        T.print_head_n(df)
        legend_dict = {}
        gez_name = df['Acronym'].tolist()
        gez_code = df['ID'].tolist()
        for i in range(len(gez_name)):
            legend_dict[gez_code[i]] = gez_name[i]
        outf = join(self.datadir,'legend')
        T.save_npy(legend_dict,outf)
        pass

class HWSD:
    def __init__(self):
        self.datadir = join(data_root, 'HWSD')
        pass

    def run(self):
        # self.gen_soil_property_dict()
        # self.gen_soil_SILT_CLAY_SAND_map()
        # self.npy_to_tif()
        # self.resample()
        # self.resample1()
        self.check_resample()
        pass

    def DEPTH_information(self):
        # see https://openknowledge.fao.org/server/api/core/bitstreams/149f1562-bf6a-439f-9d3a-eb93940f39cf/content
        # 2.3.3 Soil Attributes per depth layer
        # Page 14

        # DEPTH OF LAYER.
        # D1 = 0-20 cm
        # D2 = 20-40 cm
        # D3 = 40-60 cm
        # D4 = 60-80 cm
        # D5 = 80-100 cm
        # D6 = 100-150 cm
        # D7 = 150-200 cm
        pass

    def read_Database(self):
        fpath = join(self.datadir,'DB/HWSD2_LAYERS.txt')
        header_f = join(self.datadir,'DB/header.txt')
        hearder_list = open(header_f).readline().split()
        # print(hearder_list);exit()
        df = pd.read_csv(fpath,header=None,low_memory=False)
        df = df.rename(columns=dict(zip(range(len(hearder_list)), hearder_list)))
        return df

        pass

    def gen_soil_SILT_CLAY_SAND_map(self):
        soil_prop_df = self.gen_soil_property_dict()
        soil_prop_dict = T.df_to_dic(soil_prop_df, 'SMU')
        # for smu in soil_prop_dict:
        #     print(smu,soil_prop_dict[smu]);exit()

        # T.print_head_n(soil_prop_dict);exit()
        soil_property = ['SILT','CLAY','SAND']
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir)

        raster_f = join(self.datadir,'raster/HWSD2.bil')
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(raster_f)
        array = np.array(array, dtype=float)
        array[array>60000] = np.nan
        for p in soil_property:
            # exit()
            outpath_soil = join(outdir, p + '.npy')
            if isfile(outpath_soil):
                continue

            array_soil = np.ones_like(array)*np.nan
            for r in tqdm(range(array.shape[0]),desc=p):
                for c in range(array.shape[1]):
                    SMU = array[r,c]
                    if np.isnan(SMU):
                        continue
                    SMU = int(SMU)
                    if not SMU in soil_prop_dict:
                        continue
                    # if SMU > 60000:
                    #     continue
                    try:
                        val = soil_prop_dict[SMU][p]
                    except Exception as e:
                        val = np.nan
                        print(e)
                    array_soil[r,c] = val
            T.save_npy(array_soil,outpath_soil)
            # ToRaster().array2raster(outpath_soil, originX, originY, pixelWidth, pixelHeight, array_soil)

        pass

    def npy_to_tif(self):
        outdir = join(self.datadir,'tif')
        for f in tqdm(T.listdir(outdir)):
            fpath = join(outdir,f)
            if not f.endswith('.npy'):
                continue
            outpath = join(outdir,f.replace('.npy','.tif'))
            array = np.load(fpath)
            originX, originY, pixelWidth, pixelHeight = -180, 90, 0.0083333333333, -0.0083333333333
            ToRaster().array2raster(outpath, originX, originY, pixelWidth, pixelHeight, array)

    def resample1(self):
        fdir = join(self.datadir,'tif')
        soil_type_list = ['SILT','CLAY','SAND']
        outdir = join(self.datadir,'tif_05')
        T.mk_dir(outdir)
        for soil_type in soil_type_list:
            print(soil_type)
            fpath = join(fdir,soil_type + '.tif')
            outpath = join(outdir,soil_type + '.tif')
            if isfile(outpath):
                continue
            ToRaster().resample_reproj(fpath,outpath,0.5)

    def resample(self):
        fdir = join(self.datadir,'tif')
        soil_type_list = ['SILT','CLAY','SAND']
        outdir = join(self.datadir,'tif_05_1')
        T.mk_dir(outdir)
        for soil_type in soil_type_list:
            print(soil_type)
            fpath = join(fdir,soil_type + '.tif')
            outpath = join(outdir,soil_type + '.tif')
            if isfile(outpath):
                continue
            array = ToRaster().raster2array(fpath)[0]
            target_res = 0.5
            original_res = 0.0083333333333
            array_res = T.resample_nan(array, target_res, original_res,)
            originX, originY, pixelWidth, pixelHeight = -180, 90, 0.5, -0.5
            ToRaster().array2raster(outpath, originX, originY, pixelWidth, pixelHeight, array_res)

    def check_resample(self):
        fdir = join(self.datadir,'tif_05')
        spatial_dicts = {}
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dicts[f.split('.')[0]] = spatial_dict
        df = T.spatial_dics_to_df(spatial_dicts)
        df['sum'] = df['SILT'] + df['CLAY'] + df['SAND']
        sum_spatial_dict = T.df_to_spatial_dic(df,'sum')
        arr_sum = DIC_and_TIF().pix_dic_to_spatial_arr(sum_spatial_dict)
        plt.imshow(arr_sum,cmap='jet',vmin=99,vmax=101,interpolation='nearest')
        plt.show()


    def gen_soil_property_dict(self):
        outdir = join(self.datadir,'DB')
        outf = join(outdir,'soil_property_dict.df')
        if isfile(outf):
            df = T.load_df(outf)
            return df
        picked_cols = ['HWSD2_SMU_ID','SILT','CLAY','SAND','LAYER']
        layers_list = ['D1','D2','D3','D4','D5']
        soil_property = ['SILT','CLAY','SAND']

        df = self.read_Database()
        df = df[picked_cols]
        df_group = T.df_groupby(df,'HWSD2_SMU_ID')
        soil_property_dict = {}
        for SMU in tqdm(df_group):
            df_i = df_group[SMU]
            df_i_picked_layers = df_i[df_i['LAYER'].isin(layers_list)]
            # T.print_head_n(df_i_picked_layers)
            soil_property_dict_i = {}
            for p in soil_property:
                vals = df_i_picked_layers[p].tolist()
                vals = np.array(vals)
                vals = vals[vals>=0]
                if len(vals)<5:
                    continue
                vals_mean = np.nanmean(vals)
                soil_property_dict_i[p] = vals_mean
            SMU = int(SMU)
            soil_property_dict[SMU] = soil_property_dict_i
        df_result = T.dic_to_df(soil_property_dict,'SMU')
        # T.print_head_n(df_result);exit()
        T.save_df(df_result,outf)
        T.df_to_excel(df_result, outf)
        return df_result

        pass

def main():
    # GIMMS_NDVI().run()
    # SPEI().run()
    # SPI().run()
    # TMP().run()
    # TMX().run()
    # Precipitation().run()
    # VPD().run()
    # CCI_SM().run()
    # ERA_SM().run()
    # Terraclimate().run()
    # GLC2000().run()
    # CCI_SM().run()
    # CCI_SM_v7().run()
    # VOD_Kband().run()
    # VOD_AMSRU().run()
    # CSIF().run()
    # Terraclimate().run()
    # SPI().run()
    # GLEAM().run()
    # ERA_2m_T().run()
    # ERA_T2m_daily().run()
    # ERA_Precip().run()
    # GPCC().run()
    # BEST().run()
    # GOME2_SIF().run()
    # MODIS_LAI_Yuan().run()
    # MODIS_LAI_Chen().run()
    # FAPAR().run()
    # Aridity_Index().run()
    # TCSIF().run()
    # Global_Ecological_Zone().run()
    # IPCC_cliamte_zone().run()
    HWSD().run()

    pass




if __name__ == '__main__':
    main()
