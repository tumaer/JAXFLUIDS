import os
from typing import Dict, List, Tuple

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 14})
TITLE_FONTSIZE = 14

#TODO check planes in 3d create figure

def create_1D_animation(
        data_dict: Dict[str, np.ndarray],
        cell_centers: Tuple[np.ndarray],
        nrows_ncols: Tuple[int, int],
        times: np.ndarray | None = None,
        axis: str = "x",
        axis_values: Tuple = (0.0,0.0),
        minmax_list: List | None = None,
        save_anim: str | None = None,
        fig_args: Dict | None = None,
        axes_args: Dict | None = None,
        line_args: Dict | None = None,
        interval: int = 50,
        save_png: str | None = None,
        is_return: bool = False,
        dpi: int = 100
    ) -> Tuple[mpl.figure.Figure, FuncAnimation] | None:
    """Create a 1D line animation or save each snapshot as a PNG.

    If ``times`` is provided, the current time is shown as a dynamic
    suptitle. Returns ``(fig, ani)`` when ``is_return`` is true.
    """

    if fig_args is None:
        fig_args = {}
    if axes_args is None:
        axes_args = {}
    if line_args is None:
        line_args = {}

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
            fig, axes, *_ = create_1D_figure(
                data_dict=data_dict,
                cell_centers=X0,
                times=times,
                nrows_ncols=nrows_ncols,
                minmax_list=minmax_list,
                index=index_time,
                fig_args=fig_args,
                line_args=line_args,
                is_show=False
            )
            print("Saving %s" % filename)
            fig.savefig(os.path.join(save_png, filename), dpi=dpi, bbox_inches="tight")
            plt.clf()
            plt.cla()
            plt.close(fig)
            del fig, axes

    else:
        fig, axes, lines_list, suptitle = create_1D_figure(
            data_dict=data_dict,
            cell_centers=X0,
            times=times,
            nrows_ncols=nrows_ncols,
            minmax_list=minmax_list,
            index=0,
            fig_args=fig_args,
            line_args=line_args,
            is_show=False
        )
        ani = FuncAnimation(
            fig, update_1D,
            frames=no_timesnapshots,
            fargs=(data_dict, times, lines_list, suptitle),
            interval=interval,
            # Figure-level suptitles are not reliably redrawn with blitting.
            blit=suptitle is None,
            repeat=True
        )

        if is_return:
            return fig, ani

        save_animation(ani, save_anim, dpi)
        plt.show()

def create_2D_animation(
        data_dict: Dict[str, np.ndarray],
        cell_centers: List,
        times: np.ndarray | None = None,
        levelset: np.ndarray | None = None,
        nrows_ncols: Tuple[int, int] | None = None,
        plane: str = "xy",
        plane_value: float = 0.0,
        minmax_list: List | None = None,
        cmap: str = "seismic",
        norm: mpl.colors.Normalize | None = None,
        colorbars: Tuple[str] | None = None,
        save_anim: str | None = None,
        interval: int = 50,
        dpi: int = 100,
        fig_args: Dict | None = None,
        axes_args: Dict | None = None,
        levelset_kwargs: Dict | None = None,
        save_png: str | None = None,
        index_offset_name: int = 0,
        is_return: bool = False,
    ) -> Tuple[mpl.figure.Figure, FuncAnimation] | None:
    """Create a 2D pcolormesh animation or save each snapshot as a PNG.

    Four-dimensional data is sliced by ``plane`` and ``plane_value``.
    Returns ``(fig, ani)`` when ``is_return`` is true.
    """

    if fig_args is None:
        fig_args = {}
    if axes_args is None:
        axes_args = {}
    if levelset_kwargs is None:
        levelset_kwargs = {}

    (X0,X1), data_dict, levelset = prepare_data_2D(
        data_dict = data_dict,
        cell_centers = cell_centers,
        plane = plane,
        plane_value = plane_value,
        levelset = levelset,
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
                levelset_kwargs=levelset_kwargs,
                cmap=cmap,
                norm=norm,
                is_show=False)
            print("Saving %s" % filename)
            fig.savefig(os.path.join(save_png, filename), dpi=dpi, bbox_inches="tight")
            plt.clf()
            plt.cla()
            plt.close(fig)
            del fig, axes

    else:
        # CREATE SUBPLOTS
        fig, axes, quadmesh_list, pciset_list, suptitle = create_2D_figure(
            data_dict=data_dict,
            times=times,
            levelset=levelset,
            nrows_ncols=nrows_ncols,
            meshgrid=(X0, X1),
            minmax_list=minmax_list,
            colorbars=colorbars,
            index=0,
            cmap=cmap,
            norm=norm,
            fig_args=fig_args,
            levelset_kwargs=levelset_kwargs,
            is_show=False)

        # CREATE ANIMATION
        ani = FuncAnimation(
            fig, update_2D,
            frames=no_timesnapshots,
            fargs=((X0, X1), data_dict, levelset, times, axes, quadmesh_list, pciset_list, suptitle),
            interval=interval,
            # Figure-level suptitles are not reliably redrawn with blitting.
            blit=suptitle is None,
            repeat=True
        )
        if is_return:
            return fig, ani

        save_animation(ani, save_anim, dpi)
        plt.show()

def create_3D_animation(
        data_dict: Dict[str, np.ndarray],
        cell_centers: List,
        nrows_ncols: Tuple[int, int],
        minmax_list: List | None = None,
        cmap: str = "seismic",
        save_anim: str | None = None,
        interval: int = 100,
        save_png: str | None = None,
        index_offset_name: int = 0,
        dpi: int = 100,
        fig_args: Dict | None = None,
        axes_args: Dict | None = None,
    ) -> None:
    """Create a 3D contour-slice animation or save each snapshot as a PNG."""

    if fig_args is None:
        fig_args = {}
    if axes_args is None:
        axes_args = {}

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
        times: np.ndarray | None = None,
        nrows_ncols: Tuple[int, int] | None = None,
        axis: str | None = None,
        axis_values: Tuple | None = None,
        minmax_list: List | None = None,
        index: int = -1,
        fig_args: Dict | None = None,
        line_args: Dict | None = None,
        save_fig: str | None = None,
        dpi: int | None = None,
        is_show: bool = True,
    ) -> Tuple[mpl.figure.Figure, np.ndarray | List, List, Text | None]:
    """Create a static 1D line figure."""

    if fig_args is None:
        fig_args = {}
    if line_args is None:
        line_args = {}

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
    # fig.tight_layout()
    if times is not None:
        title_str = r"$t = " + f"{times[index]:4.3e}" + "$"
        suptitle = fig.suptitle(title_str)
    else:
        suptitle = None
    if type(axes) != np.ndarray:
        axes = [axes]
    else:
        axes = axes.flatten()

    # PLOT DATA
    lines_list = []
    for i, (ax, quantity) in enumerate(zip(axes, data_dict.keys())):
        lines_list.append(
            ax.plot(X0, data_dict[quantity][index], label=quantity, **line_args)[0]
        )
        if minmax_list is not None:
            minmax = minmax_list[i]
        else:
            minmax = [np.min(data_dict[quantity]) * 0.9, np.max(data_dict[quantity]) * 1.1]
        ax.set_ylim(minmax)
        ax.set_title(quantity.upper(), fontsize=TITLE_FONTSIZE)

    save_figure(fig, save_fig, dpi)
    if is_show:
        plt.show()
    return fig, axes, lines_list, suptitle

def create_2D_figure(
        data_dict: Dict[str, np.ndarray],
        times: np.ndarray | None = None,
        nrows_ncols: Tuple[int, int] | None = None,
        levelset: np.ndarray | None = None,
        cell_centers: List | None = None,
        meshgrid: Tuple | None = None,
        plane: str | None = None,
        plane_value: float | None = None,
        minmax_list: List | None = None,
        index: int = -1,
        cmap: str = "seismic",
        norm: mpl.colors.Normalize | None = None,
        colorbars: Tuple[str] | None = None,
        fig_args: Dict | None = None,
        levelset_kwargs: Dict | None = None,
        save_fig: str | None = None,
        dpi: int | None = None,
        is_show: bool = True,
    ) -> Tuple[mpl.figure.Figure, np.ndarray | List, List, List, Text | None]:
    """Create a static 2D pcolormesh figure.

    Returns the figure, axes, pcolormeshes, levelset contours, and suptitle.

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
    :param minmax_list: List of value ranges for each quantity, defaults to None
    :type minmax_list: List, optional
    :param index: Time snapshot index, defaults to -1
    :type index: int, optional
    :param cmap: Colormap name or object, defaults to "seismic"
    :type cmap: str, optional
    :param fig_args: Extra keyword arguments passed to plt.subplots
    :type fig_args: Dict, optional
    :param levelset_kwargs: Custom arguments which are passed to ax.contour
        when plotting the level-set contour, defaults to None
    :type levelset_kwargs: Dict, optional
    :param save_fig: Path to save the figure, defaults to None
    :type save_fig: str, optional
    :param dpi: Resolution in dpi, defaults to None
    :type dpi: int, optional
    :return: Figure, axes, pcolormeshes, levelset contours, and suptitle
    :rtype: Tuple
    """
    if fig_args is None:
        fig_args = {}
    if cell_centers is None and meshgrid is None:
        raise TypeError("Create_figure requires either cell_centers or meshgrid argument.")
   
    (X0,X1), data_dict, levelset = prepare_data_2D(
        data_dict,
        cell_centers,
        meshgrid,
        plane,
        plane_value,
        levelset,
    )
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

    # NORMS
    norms = norm
    if isinstance(norms, mpl.colors.Normalize):
        norms = [norms] * n_plots
    elif isinstance(norms, (List, Tuple)):
        for norm in norms:
            assert isinstance(norm, mpl.colors.Normalize)
    else:
        norms = None

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
            if norms is None:
                datadict_kwargs.append({"cmap": cmaps[i], "vmin": minmax[0], "vmax": minmax[1]})
            else:
                datadict_kwargs.append({"cmap": cmaps[i], "norm": norms[i]})

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
        suptitle = fig.suptitle(title_str)
    else:
        suptitle = None
    if type(axes) != np.ndarray:
        axes = [axes]
    else:
        axes = axes.flatten()

    quadmesh_list = []
    pciset_list = []

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
    return fig, axes, quadmesh_list, pciset_list, suptitle

def create_3D_figure(
        data_dict: Dict[str, np.ndarray],
        nrows_ncols: Tuple[int, int] | None = None,
        cell_centers: List | None = None,
        meshgrid: Tuple | None = None,
        minmax_list: List | None = None,
        index: int = -1,
        cmap: str = "seismic",
        fig_args: Dict | None = None,
        axes_args: Dict | None = None,
        save_fig: str | None = None,
        dpi: int | None = None,
        is_show: bool = True,
    ) -> Tuple[mpl.figure.Figure, np.ndarray | List, List, List, List]:
    """Create a static 3D contour-slice figure."""

    if fig_args is None:
        fig_args = {}
    if axes_args is None:
        axes_args = {}
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
   
    # Offsets define the three planes on which the contour slices are drawn.
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
        times: np.ndarray | None,
        lines: List,
        suptitle: Text | None
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

    if times is not None and suptitle is not None:
        title_str = r"$t = " + f"{times[i]:4.3e}" + "$"
        suptitle.set_text(title_str)
        list_of_collections.append(suptitle)

    return list_of_collections

def update_2D(
        i: int,
        meshgrid: Tuple,
        data_dict: Dict[str, np.ndarray],
        levelset: np.ndarray,
        times: np.ndarray | None,
        axes: np.ndarray,
        quadmesh_list: List,
        pciset_list: List,
        suptitle: Text | None,
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

    for quantity, ax, quadmesh, pciset in zip(data_dict, axes, quadmesh_list, pciset_list):
        
        # UPDATE QUADMESH
        quadmesh.set_array(data_dict[quantity][i].flatten())
        list_of_collections.append(quadmesh)

        # Contour sets are recreated because their collections are not updated
        # in-place like the pcolormesh array above.
        if type(levelset) != type(None) and type(pciset[0]) != type(None):
            for tp in pciset[0].collections:
                tp.remove()
            pciset[0] = ax.contour(X0, X1, np.squeeze(levelset[i]), levels=[0.0], colors="black", linewidths=2)
            list_of_collections += pciset[0].collections

    if times is not None and suptitle is not None:
        title_str = r"$t = " + f"{times[i]:4.3e}" + "$"
        suptitle.set_text(title_str)
        list_of_collections.append(suptitle)

    return list_of_collections

def update_3D(
        i: int,
        meshgrid: Tuple,
        data_dict: Dict[str, np.ndarray],
        axes: List,
        collection_list: List,
        offsets: List,
        kwargs: Dict,
    ) -> List:
    X0, X1, X2 = meshgrid

    list_of_collections = []

    for quantity, ax, collection, kw in zip(data_dict, axes, collection_list, kwargs):
        
        # Rebuild contour sets because contourf collections do not expose a
        # simple array update path comparable to pcolormesh.
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

def save_figure(fig: mpl.figure.Figure,
                save_fig: str | None = None,
                dpi: int | None = None) -> None:
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

def save_animation(ani: FuncAnimation,
                   save_anim: str | None = None,
                   dpi: int | None = None) -> None:
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
        cell_centers: List | Tuple[np.ndarray, ...],
        axis: str | None,
        axis_values: Tuple | None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

    is_3D_data = (axis is not None and axis_values is not None)
    if is_3D_data:
        # Select the nearest point on the two transverse axes, leaving one
        # temporal and one spatial dimension for each quantity.
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
        cell_centers: List | None = None,
        meshgrid: Tuple | None = None,
        plane: str | None = None,
        plane_value: float | None = None,
        levelset: np.ndarray | None = None,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, np.ndarray], np.ndarray | None]:

    axis_index = {"x": 0, "y": 1, "z": 2}

    is_3D_data = (plane is not None and plane_value is not None)
    if is_3D_data:
        # Slice the 3D volume at the nearest coordinate normal to the requested
        # plane, preserving time plus the two in-plane spatial dimensions.
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
