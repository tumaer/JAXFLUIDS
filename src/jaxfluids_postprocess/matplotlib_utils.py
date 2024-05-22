import os
from typing import Dict, List, Tuple

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 14})
TITLE_FONTSIZE = 14

#TODO do we want to pass times argument to animation??? 
#TODO check planes in 3d create figure

def create_1D_animation(
        data_dict: Dict[str, np.ndarray],
        cell_centers: Tuple[np.ndarray],
        times: np.ndarray,
        nrows_ncols: Tuple,
        axis: str = "x",
        axis_values: Tuple = (0.0,0.0),
        minmax_list: List = None,
        save_anim: str = None,
        fig_args: Dict = {},
        axes_args: Dict = {},
        interval: int = 50,
        save_png: str = None,
        dpi: int = 100
        ) -> None:
    """Creates a subplot of 1D line plots. If the levelset argument is 
    provided, the interface will be visualized as an isoline. If the 
    static_time argument is provided, no animation is created,
    but a plot at this time will be shown.

    :param data_dict: Data buffers
    :type data_dict: List
    :param cell_centers: Cell center coordinates
    :type cell_centers: List
    :param times: Time buffer
    :type times: np.ndarray
    :param nrows_ncols: Shape of subplots
    :type nrows_ncols: Tuple
    :param axis: Axis to slice, defaults to "x"
    :type axis: str, optional
    :param values: Values of remaining axis to slice, defaults to [0.0,0.0]
    :type values: List, optional
    :param save_fig: Path to save static plot, defaults to None
    :type save_fig: str, optional
    :param save_anim: Path to save animation, defaults to None
    :type save_anim: str, optional
    :param interval: Interval in ms for animation, defaults to 50
    :type interval: int, optional
    :param dpi: Resolution in dpi, defaults to 100
    :type dpi: int, optional
    """

    X0, data_dict = prepare_data_1D(
        data_dict,
        cell_centers,
        axis,
        axis_values
    )

    no_timesnapshots = len(data_dict[list(data_dict.keys())[0]])

    if save_png:
        assert_string = (
            f"The provided path {os.path.abspath(save_png)} for save_png = '{save_png}' does not exist. "
            "Please create this folder such that output can be written.")
        assert os.path.exists(save_png), assert_string

        for index_time in range(no_timesnapshots):
            filename = "image_%04d" % (index_time)
            fig, axes, _ = create_1D_figure(
                data_dict=data_dict,
                cell_centers=X0,
                nrows_ncols=nrows_ncols,
                minmax_list=minmax_list,
                index=index_time,
                fig_args=fig_args,
                is_show=False
            )
            print("Saving %s" % filename)
            fig.savefig(os.path.join(save_png, filename), dpi=dpi, bbox_inches="tight")
            plt.clf()
            plt.cla()
            plt.close(fig)
            del fig, axes

    else:
        fig, axes, lines_list = create_1D_figure(
            data_dict=data_dict,
            cell_centers=X0,
            nrows_ncols=nrows_ncols,
            minmax_list=minmax_list,
            index=0,
            fig_args=fig_args,
            is_show=False
        )
        ani = FuncAnimation(
            fig, update_1D,
            frames=no_timesnapshots,
            fargs=(data_dict, lines_list),
            interval=interval,
            blit=True,
            repeat=True
        )
        save_animation(ani, save_anim, dpi)
        plt.show()

def create_2D_animation(
        data_dict: Dict[str, np.ndarray],
        cell_centers: List,
        times: np.ndarray = None,
        levelset: np.ndarray = None,
        nrows_ncols: Tuple = None,
        plane: str = "xy",
        plane_value: float = 0.0,
        minmax_list: List = None,
        cmap: str = "seismic",
        colorbars: Tuple[str] = None,
        save_anim: str = None,
        interval: int = 50,
        dpi: int = 100,
        fig_args: Dict = {},
        axes_args: Dict = {},
        save_png: str = None,
        index_offset_name: int = 0,
        ) -> None:
    """ Standalone function which creates an animation of the data provided
    in data_dict. create_2D_figure is invoked to create a 2D pcolormesh.
    Use create_2D_figure if you want to visualize a static plot. 
    If the levelset argument is provided, the interface is
    plotted as an isoline. Returns the figure, the axes and
    the corresponding plt.pcolormesh and plt.contour objects. 

    There are 2 ways how data can be provided:
    1) Data_dict has 4-dimensional entries with (time, axis0, axis1, axis2)
        shape. Then plane ("xy", "xz", "yz") and plane_value arguments
        as well as cell_centers with 3 entries must be provided. The original
        4D buffers are sliced appropriately and a meshgrid is generated. (Default)
    
    2) Data_dict has 3-dimensional entries with (time, axis0, axis1) shape and 
        cell-centers (2 entries) are provided. Then an appropriate meshgrid is 
        generated.

    :param data_dict: [description]
    :type data_dict: Dict
    :param cell_centers: [description]
    :type cell_centers: List
    :param times: [description]
    :type times: np.ndarray
    :param levelset: [description], defaults to None
    :type levelset: np.ndarray, optional
    :param nrows_ncols: [description], defaults to None
    :type nrows_ncols: Tuple, optional
    :param plane: [description], defaults to "xy"
    :type plane: str, optional
    :param plane_value: [description], defaults to 0.0
    :type plane_value: float, optional
    :param minmax_list: [description], defaults to None
    :type minmax_list: List, optional
    :param cmap: [description], defaults to "seismic"
    :type cmap: str, optional
    :param save_anim: [description], defaults to None
    :type save_anim: str, optional
    :param interval: [description], defaults to 50
    :type interval: int, optional
    :param dpi: [description], defaults to 100
    :type dpi: int, optional
    :param fig_args: [description], defaults to {}
    :type fig_args: Dict, optional
    :param axes_args: [description], defaults to {}
    :type axes_args: Dict, optional
    :param save_png: [description], defaults to None
    :type save_png: str, optional
    :param index_offset_name: [description], defaults to 0
    :type index_offset_name: int, optional
    :return: [description]
    :rtype: [type]
    """

    (X0,X1), data_dict, levelset = prepare_data_2D(
        data_dict = data_dict,
        cell_centers = cell_centers,
        plane = plane,
        plane_value = plane_value,
        levelset = levelset
        )

    no_timesnapshots = len(data_dict[list(data_dict.keys())[0]])
    if save_png:
        assert_string = (
            f"The provided path {os.path.abspath(save_png)} for save_png = '{save_png}' does not exist. "
            "Please create this folder such that output can be written.")
        assert os.path.exists(save_png), assert_string
    
        # LOOP OVER TIME SNAPSHOTS
        for index_time in range(no_timesnapshots):
            filename = "image_%04d" % (index_time + index_offset_name)
            fig, axes, _, _, _ = create_2D_figure(
                data_dict=data_dict,
                times=times,
                levelset=levelset,
                nrows_ncols=nrows_ncols,
                meshgrid=(X0, X1),
                minmax_list=minmax_list,
                colorbars=colorbars,
                index=index_time,
                fig_args=fig_args,
                cmap=cmap,
                is_show=False)
            print("Saving %s" % filename)
            fig.savefig(os.path.join(save_png, filename), dpi=dpi, bbox_inches="tight")
            plt.clf()
            plt.cla()
            plt.close(fig)
            del fig, axes

    else:
        # CREATE SUBPLOTS
        fig, axes, quadmesh_list, pciset_list, scatter_list = create_2D_figure(
            data_dict=data_dict,
            times=times,
            levelset=levelset,
            nrows_ncols=nrows_ncols,
            meshgrid=(X0, X1),
            minmax_list=minmax_list,
            colorbars=colorbars,
            index=0,
            cmap=cmap,
            fig_args=fig_args,
            is_show=False)

        # CREATE ANIMATION
        ani = FuncAnimation(
            fig, update_2D,
            frames=no_timesnapshots,
            fargs=((X0, X1), data_dict, levelset, axes, quadmesh_list, pciset_list, scatter_list),
            interval=interval,
            blit=True,
            repeat=True
        )
        save_animation(ani, save_anim, dpi)
        plt.show()

def create_3D_animation(
        data_dict: Dict[str, np.ndarray],
        cell_centers: List,
        nrows_ncols: Tuple,
        minmax_list: List = None,
        cmap: str = "seismic",
        save_anim: str = None,
        interval: int = 100,
        save_png: str = None,
        index_offset_name = 0,
        dpi: int = 100,
        fig_args: Dict = {},
        axes_args: Dict = {},
    ) -> None:
    """Creates 3D animation. Plots contours for material fields.
    Optional save_png argument specifies a path to save
    timesnapshots as .png files.

    :param data_dict: _description_
    :type data_dict: Dict
    :param cell_centers: _description_
    :type cell_centers: List
    :param nrows_ncols: _description_
    :type nrows_ncols: Tuple
    :param minmax_list: _description_, defaults to None
    :type minmax_list: List, optional
    :param cmap: _description_, defaults to "seismic"
    :type cmap: str, optional
    :param interval: _description_, defaults to 50
    :type interval: int, optional
    :param save_png: _description_, defaults to None
    :type save_png: str, optional
    :param index_offset_name: _description_, defaults to 0
    :type index_offset_name: int, optional
    :param dpi: _description_, defaults to 100
    :type dpi: int, optional
    """

    no_timesnapshots = len(data_dict[list(data_dict.keys())[0]])
    X0, X1, X2 = np.meshgrid(*cell_centers, indexing="ij")

    if save_png:
        assert_string = (
            f"The provided path {os.path.abspath(save_png)} for save_png = '{save_png}' does not exist. "
            "Please create this folder such that output can be written.")
        assert os.path.exists(save_png), assert_string

        for index_time in range(no_timesnapshots):
            filename = "image_%04d" % (index_time + index_offset_name)
            fig, axes, _, _, _ = create_3D_figure(
                data_dict=data_dict,
                meshgrid=(X0, X1, X2),
                nrows_ncols=nrows_ncols,
                minmax_list=minmax_list,
                cmap=cmap,
                index=index_time,
                fig_args=fig_args,
                axes_args=axes_args,
                is_show=False,
            )
            print("Saving %s" % filename)
            fig.savefig(os.path.join(save_png, filename), dpi=dpi, bbox_inches="tight")
            plt.clf()
            plt.cla()
            plt.close(fig)
        
    else:
        # CREATE FIGURE
        fig, axes, collection_list, kwargs, offsets = create_3D_figure(
            data_dict=data_dict,
            meshgrid=(X0, X1, X2),
            nrows_ncols=nrows_ncols,
            minmax_list=minmax_list,
            cmap=cmap,
            fig_args=fig_args,
            axes_args=axes_args,
            is_show=False,
        )
        # CREATE ANIMATION
        ani = FuncAnimation(
            fig, update_3D,
            frames=no_timesnapshots,
            fargs=((X0, X1, X2), data_dict, axes, collection_list, offsets, kwargs),
            interval=interval,
            blit=False,
            repeat=True
        )
        save_animation(ani, save_anim, dpi)
        plt.show()

def create_1D_figure(
        data_dict: Dict[str, np.ndarray],
        cell_centers: List,
        nrows_ncols: Tuple = None,
        axis: str = None,
        axis_values: Tuple = None,
        minmax_list: List = None,
        index: int = -1,
        fig_args: Dict = {},
        save_fig: str = None,
        dpi: int = None,
        is_show: bool = True,
    ):

    X0, data_dict = prepare_data_1D(
        data_dict,
        cell_centers,
        axis,
        axis_values,
    )

    # ROWS AND COLUMNS
    if nrows_ncols is None:
        nrows_ncols = (1,len(data_dict.keys()))
    assert np.prod(nrows_ncols) == len(data_dict.keys()), \
        "Amount of requested axes and available quantities do not match."

    # CHECK DIMENSIONALITY OF DATA DICT
    for quantity, data in data_dict.items():
        assert data.ndim == 2,\
            f"Dimensionality of {quantity} has to be 2 (time, x0) but is {data.ndim}"

    # CREATE SUBFIGURE
    fig, axes = plt.subplots(
        nrows=nrows_ncols[0], 
        ncols=nrows_ncols[1],
        sharex=True, 
        squeeze=False, 
        **fig_args
    )
    fig.tight_layout()
    if type(axes) != np.ndarray:
        axes = [axes]
    else:
        axes = axes.flatten()

    # PLOT DATA
    lines_list = []
    for i, (ax, quantity) in enumerate(zip(axes, data_dict.keys())):
        lines_list.append(
            ax.plot(X0, data_dict[quantity][index], "o", label=quantity)[0]
        )
        if minmax_list is not None:
            minmax = minmax_list[i]
        else:
            minmax = [np.min(data_dict[quantity]), np.max(data_dict[quantity])]
        ax.set_ylim(minmax)
        ax.set_title(quantity.upper(), fontsize=TITLE_FONTSIZE)

    save_figure(fig, save_fig, dpi)
    if is_show:
        plt.show()
    return fig, axes, lines_list

def create_2D_figure(
        data_dict: Dict[str, np.ndarray],
        times: np.ndarray = None,
        nrows_ncols: Tuple = None,
        levelset: np.ndarray = None,
        cell_centers: List = None,
        meshgrid: Tuple = None,
        plane: str = None,
        plane_value: float = None,
        minmax_list: List = None,
        index: int = -1,
        cmap: str = "seismic",
        colorbars: Tuple[str] = None,
        fig_args: Dict = {},
        levelset_kwargs: Dict = None,
        save_fig: str = None,
        dpi: int = None,
        is_show: bool = True,
    ) -> Tuple[mpl.figure.Figure, List, List, List, List]:
    """Standalone function which creates a 2D pcolormesh of the data provided
    in data_dict. If the levelset argument is provided, the interface is
    plotted as an isoline. Returns the figure, the axes and
    the corresponding plt.pcolormesh and plt.contour objects. 

    Returns the figure an array of axes, list of pcolormeshes,
    list of contours and a list of scatter artists.

    There are 3 ways how data can be provided:
    1) Data_dict has 4-dimensional entries with (time, axis0, axis1, axis2)
        shape. Then plane ("xy", "xz", "yz") and plane_value arguments
        as well as cell_centers with 3 entries must be provided. The original
        4D buffers are sliced appropriately and a meshgrid is generated.
    
    2) Data_dict has 3-dimensional entries with (time, axis0, axis1) shape and 
        cell-centers (2 entries) are provided. Then an appropriate meshgrid is 
        generated.

    3) Data_dict has 3-dimensional entries with (times, axis0, axis1) shape and 
        meshgrid are provided.

    :param data_dict: Data dictionary: keys denote the physical
        quantities and values are the data buffers
    :type data_dict: Dict[str, np.ndarray]
    :param nrows_ncols: Shape of the subplot
    :type nrows_ncols: Tuple
    :param levelset: Level-set buffer, defaults to None
    :type levelset: np.ndarray, optional
    :param cell_centers: List of cell-center vectors, can contain 2 or 3
        elements, defaults to None
    :type cell_centers: List, optional
    :param meshgrid: Meshgrid, defaults to None
    :type meshgrid: Tuple, optional
    :param plane: String which denotes the plane to be plotted. 
        Can only be used in combination with plane_value, defaults to None
    :type plane: str, optional
    :param plane_value: Float which denotes the value of the remaining axis.
        Can only be used in combination with plane, defaults to None
    :type plane_value: float, optional
    :param minmax_list: [description], defaults to None
    :type minmax_list: List, optional
    :param index: [description], defaults to -1
    :type index: int, optional
    :param cmap: [description], defaults to "seismic"
    :type cmap: str, optional
    :param fig_args: [description], defaults to {}
    :type fig_args: Dict, optional
    :param levelset_kwargs: Custom arguments which are passed to ax.contour
        when plotting the level-set contour, defaults to None
    :type levelset_kwargs: Dict, optional
    :param save_fig: [description], defaults to None
    :type save_fig: str, optional
    :param dpi: [description], defaults to None
    :type dpi: int, optional
    :return: [description]
    :rtype: Tuple
    """
    if cell_centers is None and meshgrid is None:
        raise TypeError("Create_figure requires either cell_centers or meshgrid argument.")
   
    (X0,X1), data_dict, levelset = prepare_data_2D(
        data_dict,
        cell_centers,
        meshgrid,
        plane,
        plane_value,
        levelset)
    n_plots = len(data_dict.keys())

    # ROWS AND COLUMNS
    if nrows_ncols is None:
        nrows_ncols = (1,n_plots)
    assert np.prod(nrows_ncols) == n_plots, \
        "Amount of requested axes and available quantities do not match."

    # CHECK DIMENSIONALITY OF DATA DICT
    for quantity, data in data_dict.items():
        assert data.ndim == 3,\
            f"Dimensionality of {quantity} has to be 3 (time, x0, x1) but is {data.ndim}"
    
    # COLORBARS
    if colorbars in ("horizontal", "vertical"):
        colorbars = [colorbars] * n_plots
    if colorbars is not None:
        assert len(colorbars) == n_plots, \
            "Provided number of colorbar settings is not equal to number of plots."
    else:
        colorbars = [False] * n_plots

    # CMAPS
    cmaps = cmap
    if isinstance(cmaps, (str, mpl.colors.Colormap)):
        cmaps = [cmaps] * n_plots
    if cmaps is not None:
        assert len(cmaps) == n_plots, \
            "Provided number of cmaps settings is not equal to number of plots."
    else:
        cmaps = [None] * n_plots

    # KWARGS FOR PCOLORMESH - MINMAX VALUE AND CMAP
    datadict_kwargs = []
    for i, (quantity, data) in enumerate(data_dict.items()):
        if minmax_list != None:
            minmax = minmax_list[i]
        else:
            minmax = [np.min(data), np.max(data)]

        if quantity == "schlieren":
            datadict_kwargs.append({
                "cmap": "binary", 
                "norm": mpl.colors.LogNorm(vmin=minmax[0], vmax=minmax[1])
            })
        else:
            datadict_kwargs.append({"cmap": cmaps[i], "vmin": minmax[0], "vmax": minmax[1]})
    # KWARGS FOR LEVELSET
    if levelset_kwargs is None:
        levelset_kwargs = {"colors": "black", "linewidths": 2}

    # CREATE SUBFIGURE
    fig, axes = plt.subplots(
        nrows=nrows_ncols[0], 
        ncols=nrows_ncols[1],
        sharex=True, 
        sharey=True, 
        squeeze=False, 
        **fig_args)
    fig.tight_layout()
    if times is not None:
        title_str = r"$t = " + f"{times[index]:4.3e}" + "$"
        fig.suptitle(title_str)
    if type(axes) != np.ndarray:
        axes = [axes]
    else:
        axes = axes.flatten()

    quadmesh_list = []
    pciset_list = []
    scatter_list = []

    # LOOP OVER QUANTITIES
    for i, (ax, quantity, datadict_kw) in enumerate(zip(axes, data_dict.keys(), datadict_kwargs)):
        # PCOLORMESH
        quadmesh = ax.pcolormesh(
            X0, X1, data_dict[quantity][index],
            shading="auto", **datadict_kw)
        if colorbars[i]:
            if colorbars[i] == "horizontal":
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('bottom', size='5%', pad=0.05)
                fig.colorbar(quadmesh, cax=cax, orientation='horizontal')
                
            elif colorbars[i] == "vertical":
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(quadmesh, cax=cax, orientation='vertical')

        quadmesh_list.append(quadmesh)
        # ZERO LEVELSET
        if type(levelset) != type(None):
            pciset = [ax.contour(X0, X1, levelset[index], levels=[0.0], **levelset_kwargs)]
            pciset_list.append(pciset)
        else:
            pciset_list.append([None])

        # AXIS TICKS
        ax.set_title(quantity.upper(), fontsize=TITLE_FONTSIZE, pad=10)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_aspect("equal")
        plt.tight_layout()

    save_figure(fig, save_fig, dpi)
    if is_show:
        plt.show()
    return fig, axes, quadmesh_list, pciset_list, scatter_list

def create_3D_figure(
        data_dict: Dict[str, np.ndarray],
        nrows_ncols: Tuple = None,
        cell_centers: List = None,
        meshgrid: Tuple = None,
        minmax_list = None,
        index: int = -1,
        cmap: str = "seismic",
        fig_args: Dict = {},
        axes_args: Dict = {},
        save_fig: str = None,
        dpi: int = None,
        is_show: bool = True,
    ):
    """Creates a 3d plot.

    :param data_dict: _description_
    :type data_dict: Dict
    :param cell_centers: _description_
    :type cell_centers: List
    :param nrows_ncols: _description_
    :type nrows_ncols: Tuple
    :param index: _description_, defaults to 0
    :type index: int, optional
    :param minmax_list: _description_, defaults to None
    :type minmax_list: _type_, optional
    :param cmap: _description_, defaults to "seismic"
    :type cmap: str, optional
    :return: _description_
    :rtype: _type_
    """
    if cell_centers is None and meshgrid is None:
        raise TypeError("Create_figure requires either cell_centers or meshgrid argument.")

    # GENERATE MESH GRID
    if cell_centers is not None:
        assert len(cell_centers) == 3, "create_3D_figure takes only cell_centers of length 3."
        X0, X1, X2 = np.meshgrid(*cell_centers, indexing="ij")
    else:
        X0, X1, X2 = meshgrid

    xmin, ymin, zmin = np.min(X0), np.min(X1), np.min(X2)
    xmax, ymax, zmax = np.max(X0), np.max(X1), np.max(X2)

    # CREATE SUBPLOT
    fig, axes = plt.subplots(
        nrows=nrows_ncols[0], 
        ncols=nrows_ncols[1], 
        sharex=True, 
        sharey=True, 
        subplot_kw=dict(projection="3d"),
        **fig_args,
    )
    fig.tight_layout()
    if type(axes) != np.ndarray:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # ROWS AND COLUMNS
    nrows_ncols = (1,len(data_dict.keys())) if nrows_ncols == None else nrows_ncols
    assert np.prod(nrows_ncols) == len(data_dict.keys()),\
        "Amount of requested axes and available quantities do not match."

    # KWARGS FOR PCOLORMESH - MINMAX VALUE AND CMAP
    kwargs = []
    for i, (quantity, data) in enumerate(data_dict.items()):

        if minmax_list != None:
            minmax = minmax_list[i]
        else:
            minmax = [np.min(data), np.max(data)]

        if quantity == "schlieren":
            kwargs.append({
                "cmap": "binary", 
                "norm": mpl.colors.LogNorm(vmin=minmax[0], vmax=minmax[1])
            })
        else:
            kwargs.append( {"cmap": cmap, "vmin": minmax[0], "vmax": minmax[1], "levels": 200} )
   
    if "offsets" in axes_args:
        offsets = axes_args["offsets"]
    else:
        offsets = [xmax, 0, zmax]

    collection_list = []

    # LOOP OVER QUANTITIES
    for ax, quantity, kw in zip(axes, data_dict.keys(), kwargs):
        if "view" in axes_args:
            ax.view_init(*axes_args["view"])
        if "facecolor" in axes_args:
            ax.set_facecolor(axes_args["facecolor"])
        if "dist" in axes_args:
            ax.dist = axes_args["dist"]

        index2 = np.argmin(np.abs(X2[0,0,:] - offsets[2]))
        contour0 = ax.contourf(
            X0[:,:,index2], X1[:,:,index2], data_dict[quantity][index,:,:,index2],
            zdir="z", offset=offsets[2], **kw)
        index1 = np.argmin(np.abs(X1[0,:,0] - offsets[1]))
        contour1 = ax.contourf(
            X0[:,index1,:], data_dict[quantity][index,:,index1,:], X2[:,index1,:],
            zdir="y", offset=offsets[1], **kw)
        index0 = np.argmin(np.abs(X0[:,0,0] - offsets[0]))
        contour2 = ax.contourf(
            data_dict[quantity][index,index0,:,:], X1[index0,:,:], X2[index0,:,:],
            zdir="x", offset=offsets[0], **kw)
        collection_list.append([contour0, contour1, contour2])

        ax.set_xlim3d(xmin,xmax)
        ax.set_ylim3d(ymin,ymax)
        ax.set_zlim3d(zmin,zmax)
        ax.set_aspect("equal")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_axis_off()

    save_figure(fig, save_fig, dpi)
    return fig, axes, collection_list, kwargs, offsets

def update_1D(
    i: int,
    data_dict: Dict[str, np.ndarray],
    lines: List
    ) -> List:
    """Update function for FuncAnimation for the create_1D_animation() function.

    :param i: Time snapshot index
    :type i: int
    :param data: Data buffers
    :type data: List
    :param lines: Line objects
    :type lines: List
    :return: List of collections
    :rtype: List
    """
    list_of_collections = []
    for quantity, line in zip(data_dict, lines):
        line.set_ydata(data_dict[quantity][i,:])
        list_of_collections.append(line)
    return list_of_collections

def update_2D(
    i: int,
    meshgrid: Tuple,
    data_dict: Dict[str, np.ndarray],
    levelset: np.ndarray,
    axes: np.ndarray,
    quadmesh_list: List,
    pciset_list: List,
    scatter_list: List
    ) -> List:
    """Update function for FuncAnimation for the create_2D_animation() function.

    :param i: Time snapshot index
    :type i: int
    :param X1: X1 meshgrid
    :type X1: np.ndarray
    :param X2: X2 meshgrid
    :type X2: np.ndarray
    :param data: Data buffers
    :type data: List
    :param levelset: Levelet buffer
    :type levelset: np.ndarray
    :param axes: Axes
    :type axes: np.ndarray
    :param quadmesh_list: Quadmesh objects returned by pcolormesh()
    :type quadmesh_list: List
    :param pciset_list: Pciset objects return by contour()
    :type pciset_list: List
    :return: List of collections
    :rtype: List
    """
    X0, X1 = meshgrid
    list_of_collections = []

    for quantity, ax, quadmesh, pciset, scatter in zip(data_dict, axes, quadmesh_list, pciset_list, scatter_list):
        
        # UPDATE QUADMESH
        quadmesh.set_array(data_dict[quantity][i].flatten())
        list_of_collections.append(quadmesh)

        # UPDATE ISOLINE LEVELSET
        if type(levelset) != type(None) and type(pciset[0]) != type(None):
            for tp in pciset[0].collections:
                tp.remove()
            pciset[0] = ax.contour(X0, X1, np.squeeze(levelset[i]), levels=[0.0], colors="black", linewidths=2)
            list_of_collections += pciset[0].collections

    return list_of_collections

def update_3D(
    i: int,
    meshgrid: Tuple,
    data_dict: Dict[str, np.ndarray],
    axes: List,
    collection_list: List,
    offsets: List,
    kwargs: Dict,
    ):
    X0, X1, X2 = meshgrid

    list_of_collections = []

    for quantity, ax, collection, kw in zip(data_dict, axes, collection_list, kwargs):
        
        # UPDATE QUADMESH
        for tp1, tp2, tp3 in zip(collection[0].collections, collection[1].collections, collection[2].collections):
            tp1.remove()
            tp2.remove()
            tp3.remove()

        xmin, ymin, zmin = np.min(X0), np.min(X1), np.min(X2)
        xmax, ymax, zmax = np.max(X0), np.max(X1), np.max(X2)

        index2 = np.argmin(np.abs(X2[0,0,:] - offsets[2]))
        collection[0] = ax.contourf(
            X0[:,:,index2], X1[:,:,index2], data_dict[quantity][i,:,:,index2],
            zdir="z", offset=offsets[2], **kw)
        index1 = np.argmin(np.abs(X1[0,:,0] - offsets[1]))
        collection[1] = ax.contourf(
            X0[:,index1,:], data_dict[quantity][i,:,index1,:], X2[:,index1,:],
            zdir="y", offset=offsets[1], **kw)
        index0 = np.argmin(np.abs(X0[:,0,0] - offsets[0]))
        collection[2] = ax.contourf(
            data_dict[quantity][i,index0,:,:], X1[index0,:,:], X2[index0,:,:],
            zdir="x", offset=offsets[0], **kw)

        list_of_collections += collection

    return list_of_collections

def save_figure(fig,
                save_fig: str = None,
                dpi: int = None) -> None:
    if save_fig is not None:
        dirname = os.path.abspath(os.path.dirname(save_fig))
        assert_string = (
            f"The provided parent directory {dirname} for save_fig = '{save_fig}' does not exist. "
            "Please create this folder such that output can be written.")
        assert os.path.exists(dirname), assert_string

        if save_fig.endswith((".png", ".pdf")):
            fig.savefig(save_fig, bbox_inches="tight", dpi=dpi)
        else:
            fig.savefig("%s.pdf" % save_fig, bbox_inches="tight")

def save_animation(ani,
                   save_anim: str = None,
                   dpi: int = None) -> None:
    if save_anim is not None:
        dirname = os.path.abspath(os.path.dirname(save_anim))
        assert_string = (
            f"The provided parent directory {dirname} for save_anim = '{save_anim}' does not exist. "
            "Please create this folder such that output can be written.")
        assert os.path.exists(dirname), assert_string

        if save_anim.endswith((".avi", ".mp4")):
            ani.save(save_anim, dpi=dpi)
        else:
            ani.save("%s.gif" % save_anim, dpi=dpi)

def prepare_data_1D(
        data_dict: Dict[str, np.ndarray],
        cell_centers: List,
        axis: str,
        axis_values: Tuple
        ) -> Tuple:

    is_3D_data = (axis is not None and axis_values is not None)
    if is_3D_data:
        x, y, z = cell_centers
        if axis == "x":
            X0 = x
            index1 = np.argmin(np.abs(y - axis_values[0]))
            index2 = np.argmin(np.abs(z - axis_values[1]))
            plane_slice = np.s_[:,:,index1,index2] 
        elif axis == "y":
            X0 = y
            index1 = np.argmin(np.abs(x - axis_values[0]))
            index2 = np.argmin(np.abs(z - axis_values[1]))
            plane_slice = np.s_[:,index1,:,index2]
        elif axis == "z":
            X0 = z
            index1 = np.argmin(np.abs(x - axis_values[0]))
            index2 = np.argmin(np.abs(y - axis_values[1]))
            plane_slice = np.s_[:,index1,index2,:]
        else:
            assert False, "axis does not exist"

        data_dict_1D = {}
        for quantity in data_dict.keys():
            data_dict_1D[quantity] = data_dict[quantity][plane_slice]
    else:
        data_dict_1D = data_dict
        for quantity, data in data_dict.items():
            assert data.ndim == 2, f"""If axis and axis_values are not provided,
            create_1D_figure expects data_dict entries to be 2-dimensional: one temporal
            and one spatial dimensions. However dimension is {data.ndim} for {quantity}."""
        if cell_centers is not None:
            if type(cell_centers) == list:
                assert len(cell_centers) == 1, """If axis and axis_values are not provided,
                but cell_centers is, create_1D_figure expects cell_centers of length 1."""
                meshgrid = np.meshgrid(*cell_centers, indexing="ij")
                X0 = cell_centers[0]
            else:
                X0 = cell_centers
    return X0, data_dict_1D

def prepare_data_2D(
        data_dict: Dict[str, np.ndarray],
        cell_centers: List = None,
        meshgrid = None,
        plane: str = None,
        plane_value: float = None,
        levelset: np.ndarray = None,
        ) -> Tuple:

    axis_index = {"x": 0, "y": 1, "z": 2}

    is_3D_data = (plane is not None and plane_value is not None)
    if is_3D_data:
        # SET UP SLICE OBJECTS
        remaining_axis      = [axis for axis in axis_index.keys() if axis not in plane][0]
        plane_coordinates   = [cell_centers[axis_index[axis]] for axis in plane]
        meshgrid            = np.meshgrid(*plane_coordinates, indexing="ij")
        index               = np.argmin(np.abs(cell_centers[axis_index[remaining_axis]] - plane_value))
        plane_slice         = [np.s_[:,index,:,:], np.s_[:,:,index,:], np.s_[:,:,:,index]][axis_index[remaining_axis]]

        # SLICE DATA
        data_dict_2D = {}
        for quantity in data_dict.keys():
            data_dict_2D[quantity] = data_dict[quantity][plane_slice]
        if type(levelset) != type(None):
            levelset = levelset[plane_slice]
    else:
        data_dict_2D = data_dict
        for quantity, data in data_dict.items():
            assert data.ndim == 3, f"""If plane and plane_value are not provided,
            create_2D_figure expects data_dict entries to be 3-dimensional: one temporal
            and two spatial dimensions. However dimension is {data.ndim}."""
        if cell_centers is not None:
            assert len(cell_centers) == 2, """If plane and plane_value are not provided,
            but cell_centers is, create_2D_figure expects cell_centers of length 2."""
            meshgrid = np.meshgrid(*cell_centers, indexing="ij")

    return meshgrid, data_dict_2D, levelset