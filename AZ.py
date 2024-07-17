# coding=utf-8
import matplotlib.pyplot as plt
from lytools import *
T = Tools()
this_root = '/home/liyang/Desktop/disk/Energy_water_hotdrought/AZ/'

def clip():
    fdir = join(this_root,'tif')
    outdir = join(this_root,'tif_clip')
    T.mk_dir(outdir,force=True)
    shp = join(this_root,'shp','az.shp')
    for f in T.listdir(fdir):
        fpath = join(fdir,f)
        print(fpath)
        outf = join(outdir,f)
        ToRaster().clip_array(fpath,outf,shp)
    pass


def plot_map():
    fdir = join(this_root,'az_GPP_annual_mean')
    outdir = join(this_root,'tif_clip_map_GPP')
    shp = join(this_root,'shp','az')
    T.mk_dir(outdir,force=True)

    for f in T.listdir(fdir):
        if not f.endswith('.tif'):
            continue
        fpath = join(fdir,f)
        outf = join(outdir,f+'.pdf')
        plt.figure(figsize=(5,5))
        # Plot_AZ().plot_az(fpath,shp,vmin=-1,vmax=1)
        Plot_AZ().plot_az(fpath,shp,vmin=0,vmax=0.8)
        plt.title(f)
        # plt.show()
        plt.savefig(outf)
        plt.close()
    pass

class Plot_AZ:
    def __init__(self):
        pass

    def plot_az(self, fpath, in_shpfile,ax=None, cmap=None, vmin=None, vmax=None, is_plot_colorbar=True, is_reproj=True,res=10000,is_discrete=False,colormap_n=11):
        '''
        :param fpath: tif file
        :param is_reproj: if True, reproject file from 4326 to Robinson
        :param res: resolution, meter
        '''
        color_list = [
            '#844000',
            '#fc9831',
            '#fffbd4',
            '#86b9d2',
            '#064c6c',
        ]
        # Blue represents high values, and red represents low values.
        if ax == None:
            # plt.figure(figsize=(10, 10))
            ax = plt.subplot(1, 1, 1)
        if cmap is None:
            cmap = Tools().cmap_blend(color_list)
        arr_deg, originX_deg, originY_deg, pixelWidth_deg, pixelHeight_deg = ToRaster().raster2array(fpath)
        llcrnrlon = originX_deg
        urcrnrlat = originY_deg
        urcrnrlon = originX_deg + pixelWidth_deg * arr_deg.shape[1]
        llcrnrlat = originY_deg + pixelHeight_deg * arr_deg.shape[0]
        arr_deg = Tools().mask_999999_arr(arr_deg, warning=False)
        arr_m = ma.masked_where(np.isnan(arr_deg), arr_deg)
        # exit()
        lon_list = np.arange(originX_deg, originX_deg +  pixelWidth_deg * arr_deg.shape[1], pixelWidth_deg)
        lat_list = np.arange(originY_deg, originY_deg + pixelHeight_deg * arr_deg.shape[0], pixelHeight_deg)
        lat_list = lat_list + pixelHeight_deg / 2
        lon_list = lon_list + pixelWidth_deg / 2

        m = Basemap(projection='cyl', ax=ax, resolution='i',
                    llcrnrlon=-115, llcrnrlat=31.2, urcrnrlon=-109, urcrnrlat=37.2,
                    lon_0=-111,lat_0=0,lat_1=25,lat_2=47)
        m.drawparallels(np.arange(32., 37.1, 1.), zorder=99, linewidth=1, dashes=[4, 4])
        m.drawmeridians(np.arange(-115., -108.5, 1.), zorder=99, linewidth=1, dashes=[4, 4])

        # m.drawparallels((-90., 90.), zorder=99, dashes=[1, 0], linewidth=2)
        lon_matrix = []
        lat_matrix = []
        for lon in tqdm(lon_list):
            lon_matrix_i = []
            lat_matrix_i = []
            for lat in lat_list:
                # print(lon,lat)
                lon_projtran, lat_projtran = m.projtran(lon,lat)
                lon_matrix_i.append(lon_projtran)
                lat_matrix_i.append(lat_projtran)

            lon_matrix.append(lon_matrix_i)
            lat_matrix.append(lat_matrix_i)
        lon_matrix = np.array(lon_matrix)
        lat_matrix = np.array(lat_matrix)

        ret = m.pcolormesh(lon_matrix, lat_matrix, arr_deg.T, cmap=cmap, zorder=99, vmin=vmin, vmax=vmax)
        shp_f = in_shpfile
        m.readshapefile(shp_f,'a', drawbounds=True, linewidth=1, color='gray', zorder=-9)
        # m.readshapefile(shp_provinces_f, 'ooo', drawbounds=True, linewidth=0.3, color='k', zorder=100)
        # m.drawparallels(np.arange(-60., 90., 20.), zorder=99, dashes=[8, 8], linewidth=.5)
        # m.drawparallels((-90., 90.), zorder=99, dashes=[1, 0], linewidth=2)
        # meridict = m.drawmeridians(np.arange(0., 420., 20.), zorder=100, latmax=90, dashes=[8, 8], linewidth=.5)
        # meridict = m.drawmeridians((-180,180), zorder=100, latmax=90, dashes=[1, 0], linewidth=2)
        plt.axis('off')

        # for obj in meridict:
        #     line = meridict[obj][0][0]
        # coastlines = m.drawcoastlines(zorder=100, linewidth=0.2)
        # polys = m.fillcontinents(color='#D1D1D1', lake_color='#EFEFEF',zorder=90)
        if is_plot_colorbar:
            if is_discrete:
                bounds = np.linspace(vmin, vmax, colormap_n)
                # norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                cax,kw = mpl.colorbar.make_axes(ax,location='bottom',pad=0.05,shrink=0.5)
                # cax,kw = mpl.colorbar.make_axes(ax,location='bottom',pad=0.05)
                cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, orientation='horizontal')
            else:
                cbar = plt.colorbar(ret, ax=ax, shrink=0.5, location='bottom', pad=0.05)
        return m, ret


def main():
    # clip()
    plot_map()
    pass

if __name__ == '__main__':
    main()