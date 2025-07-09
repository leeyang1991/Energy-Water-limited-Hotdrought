# coding=utf-8
import matplotlib.pyplot as plt
from lytools import *
from pprint import pprint
import xarray as xr
T = Tools()

class ESA_CCI:
    def __init__(self):
        self.data_dir = r"E:\yang\ESA-CCI_LC/"
        # self.data_dir = '/mnt/disk3/ESA_CCI_LC/'
        pass

    def run(self):
        self.aggregate_lc()
        # self.gen_tif()
        pass


    def return_variables_list(self):
        variable_list_str = '''WATER
        BARE
        BUILT
        GRASS-MAN
        GRASS-NAT
        SHRUBS-BD
        SHRUBS-BE
        SHRUBS-ND
        SHRUBS-NE
        # WATER_INLAND
        SNOWICE
        TREES-BD
        TREES-BE
        TREES-ND
        TREES-NE
        LAND'''
        variables_list = variable_list_str.replace(' ','')
        variables_list = variables_list.split('\n')
        return variables_list

    def aggregate_lc(self):
        # fdir = join(self.data_dir,'ESACCI-LC-L4-PFT-Map-300m-P1Y-1992-2020-v2.0.8')
        fdir = join(self.data_dir,'ESACCI-LC-L4-PFT-Map-300m-P1Y-1992-2020-v2.0.8')
        # print(isdir(r'D:\ESA_CCI_LC\ESACCI-LC-L4-PFT-Map-300m-P1Y-1992-2020-v2.0.8'))
        # print(fdir);exit()
        outdir = join(self.data_dir,'tif05')
        # outdir = join(self.data_dir,'df25')
        pix_size = .5
        T.mk_dir(outdir,force=True)
        params_list = []
        for f in T.listdir(fdir):
            # print(f)
            year_str = f.split('-')[-2]
            outdir_i = join(outdir,year_str)
            T.mk_dir(outdir_i,force=True)
            params = [fdir,f,outdir_i,pix_size]
            params_list.append(params)
            # self.kernel_aggregate_lc(params)
            # self.kernel_aggregate_lc_fast(params)
        MULTIPROCESS(self.kernel_aggregate_lc_fast,params_list).run(process=6)

    def kernel_aggregate_lc(self,params):
        fdir,f,outdir,pix_size = params
        fpath = join(fdir, f)
        outf = join(outdir, f + '.df')
        # if os.path.exists(outf):
        #     return

        nc_r = xr.open_dataset(fpath)
        variables_list = self.return_variables_list()
        # arr = nc_r['WATER']
        # print(arr)
        # plt.imshow(arr[0])
        # plt.show()
        lon = nc_r['lon']
        lat = nc_r['lat']
        # print(len(lon),len(lat)) # 129600 64800
        pixelWidth = 360 / len(lon)
        pixelHeight = 180 / len(lat)
        new_pixelWidth = pix_size
        new_pixelHeight = pix_size
        nx = new_pixelWidth / pixelWidth
        ny = new_pixelHeight / pixelHeight

        new_lon = np.arange(-180, 180, new_pixelWidth)
        new_lat = np.arange(90, -90, -new_pixelHeight)
        spatial_dict = {}
        for new_r in tqdm(range(len(new_lat))):
            r_range = [new_r * nx, (new_r + 1) * nx]
            r_range_int = [int(r_range[0]), int(r_range[1])]
            for new_c in range(len(new_lon)):
                c_range = [new_c * ny, (new_c + 1) * ny]
                c_range_int = [int(c_range[0]), int(c_range[1])]
                pix = (new_r, new_c)
                spatial_dict_i = {}
                for var in variables_list:
                    # spatial_dict[(new_r, new_c)] = nc_r[var][r_range[0]:r_range[1], c_range[0]:c_range[1]]
                    # print(vals)
                    try:
                        vals = nc_r[var][0][r_range_int[0]:r_range_int[1], c_range_int[0]:c_range_int[1]]
                        vals_values = vals.values
                        vals_values = np.array(vals_values, dtype=np.float32)
                        vals_mean = np.nanmean(vals_values)
                    except:
                        vals_mean = np.nan
                    spatial_dict_i[var] = vals_mean
                spatial_dict[pix] = spatial_dict_i
        df_result = T.dic_to_df(spatial_dict, 'pix')
        T.save_df(df_result, outf)
        T.df_to_excel(df_result, outf)
        spatial_dict = {}
        pass

    def kernel_aggregate_lc_fast(self,params):
        fdir,f,outdir,pix_size = params
        fpath = join(fdir, f)

        nc_r = xr.open_dataset(fpath)
        variables_list = self.return_variables_list()

        lat_factor = int(pix_size / 0.00277777777777)
        lon_factor = int(pix_size / 0.00277777777777)
        # resampled_array = data_array.coarsen(lat=lat_factor, lon=lon_factor, boundary="trim").mean()
        spatial_dicts = {}

        for var in tqdm(variables_list,desc=f):
            outf = join(outdir, f + '_' + var + '.tif')
            try:
                data_array = nc_r[var][0]
            except:
                continue
            resampled_array = data_array.coarsen(lat=lat_factor, lon=lon_factor, boundary="trim").mean()
            resampled_array = np.array(resampled_array)
            # print('dicting')
            DIC_and_TIF(pixelsize=pix_size).arr_to_tif(resampled_array,outf)

    def kernel_aggregate_lc_GPU(self,params):
        import cupy
        import cupy_xarray
        fdir,f,outdir,pix_size = params
        fpath = join(fdir, f)
        outf = join(outdir, f + '.df')
        # if os.path.exists(outf):
        #     return

        nc_r = xr.open_dataset(fpath)
        variables_list = self.return_variables_list()
        # arr = nc_r['WATER']
        # print(arr)
        # plt.imshow(arr[0])
        # plt.show()
        lon = nc_r['lon']
        lat = nc_r['lat']
        # print(len(lon),len(lat)) # 129600 64800
        pixelWidth = 360 / len(lon)
        pixelHeight = 180 / len(lat)
        new_pixelWidth = pix_size
        new_pixelHeight = pix_size
        nx = new_pixelWidth / pixelWidth
        ny = new_pixelHeight / pixelHeight

        new_lon = np.arange(-180, 180, new_pixelWidth)
        new_lat = np.arange(90, -90, -new_pixelHeight)
        spatial_dict = {}
        for new_r in tqdm(range(len(new_lat))):
            r_range = [new_r * nx, (new_r + 1) * nx]
            r_range_int = [int(r_range[0]), int(r_range[1])]
            for new_c in range(len(new_lon)):
                c_range = [new_c * ny, (new_c + 1) * ny]
                c_range_int = [int(c_range[0]), int(c_range[1])]
                pix = (new_r, new_c)
                spatial_dict_i = {}
                for var in variables_list:
                    # spatial_dict[(new_r, new_c)] = nc_r[var][r_range[0]:r_range[1], c_range[0]:c_range[1]]
                    # print(vals)
                    try:
                        vals = nc_r[var][0][r_range_int[0]:r_range_int[1], c_range_int[0]:c_range_int[1]]
                        vals_values = vals.values
                        vals_values = np.array(vals_values, dtype=np.float32)
                        vals_mean = np.nanmean(vals_values)
                    except:
                        vals_mean = np.nan
                    spatial_dict_i[var] = vals_mean
                spatial_dict[pix] = spatial_dict_i
        df_result = T.dic_to_df(spatial_dict, 'pix')
        T.save_df(df_result, outf)
        T.df_to_excel(df_result, outf)
        spatial_dict = {}
        pass

    def gen_tif(self):
        fdir = join(self.data_dir,'df05')
        outdir = join(self.data_dir,'tif05')
        T.mk_dir(outdir,force=True)
        for f in T.listdir(fdir):
            if not f.endswith('.df'):
                continue
            fpath = join(fdir,f)
            df = T.load_df(fpath)
            pix_list = df['pix'].tolist()
            # for pix in pix_list:
            #     print(pix)
            # exit()
            T.print_head_n(df)
            variable_list = self.return_variables_list()
            for var in variable_list:
                spatial_dict = T.df_to_spatial_dic(df, var)
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
                plt.imshow(arr,interpolation='nearest',vmin=0,vmax=100,cmap='jet')
                plt.colorbar()
                plt.title(var)
                plt.show()
                pass
            pass

def main():
    ESA_CCI().run()

    pass

if __name__ == '__main__':
    main()