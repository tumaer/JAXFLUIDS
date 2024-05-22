import argparse
import os

from paraview.simple import *

def get_paraview_preset(cmap):
    cmap_dict = {
        "bgo"       : "Blue - Green - Orange",
        "bbw"       : "Black, Blue and White",
        "bdw"       : "Rainbow Blended White",
        "burd"      : "BuRd",
        "coldhot"   : "Cold and Hot",
        "coolwarm"  : "Cool to Warm",
        "inferno"   : "Inferno (matplotlib)",
        "jet"       : "Jet",
        "magma"     : "Magma (matplotlib)",
        "plasma"    : "Plasma (matplotlib)",
        "rainbow"   : "Rainbow Uniform",
        "spectral"  : "Spectral_lowBlue",
        "turbo"     : "Turbo",
        "viridis"   : "Viridis (matplotlib)",
        "xray"      : "X Ray",
        "yellow15"  : "Yellow 15",
    }

    preset = cmap_dict[cmap] if cmap in cmap_dict.keys() else None
    return preset

def get_positions_dict(bounds):
    center = ((bounds[0] + bounds[1])/2, (bounds[2] + bounds[3])/2, (bounds[4] + bounds[5])/2)
    positions_dict = {
        "center"            : center, 
        "west_north_top"    : (bounds[0], bounds[3], bounds[5]),
        "west_north_bottom" : (bounds[0], bounds[3], bounds[4]),
        "west_south_top"    : (bounds[0], bounds[2], bounds[5]),
        "west_south_bottom" : (bounds[0], bounds[2], bounds[4]),
        "east_north_top"    : (bounds[1], bounds[3], bounds[5]),
        "east_north_bottom" : (bounds[1], bounds[3], bounds[4]),
        "east_south_top"    : (bounds[1], bounds[2], bounds[5]),
        "east_south_bottom" : (bounds[1], bounds[2], bounds[4]),
        "east_center"       : (bounds[1], center[1], center[2]),  
        "west_center"       : (bounds[0], center[1], center[2]),
        "north_center"      : (center[0], bounds[3], center[2]),  
        "south_center"      : (center[0], bounds[2], center[2]),
        "top_center"        : (center[0], center[1], bounds[5]),  
        "bottom_center"     : (center[0], center[1], bounds[4]),    
        }
    return positions_dict

def generate_paraview_pngs(
        files, 
        save_path, 
        naming_offset, 
        contour_keys=None, 
        contour_values=None, 
        contour_color_keys=None, 
        contour_cmaps=None, 
        contour_opacities=None, 
        contour_color_lims=None, 
        contour_speculars=None, 
        contour_opacity_mappings=None,
        slice_keys=None, 
        slice_origins=None, 
        slice_normals=None,
        slice_logs=None, 
        slice_cmaps=None, 
        slice_opacities=None, 
        slice_color_lims=None,
        slice_translations=None,
        is_orientation_axis=False,
        background_color=None,
        camera_position=None, 
        camera_focal_point=None, 
        camera_view_up=None, 
        camera_view_angle=None,
        camera_dolly=None,
        resolution=None
    ):
    """generate_paraview_pngs [summary]

    :param files: [description]
    :type files: [type]
    :param save_path: [description]
    :type save_path: [type]
    :param contour_keys: [description], defaults to None
    :type contour_keys: [type], optional
    :param contour_values: [description], defaults to None
    :type contour_values: [type], optional
    :param contour_color_keys: [description], defaults to None
    :type contour_color_keys: [type], optional
    :param contour_opacities: [description], defaults to None
    :type contour_opacities: [type], optional
    :param contour_color_lims: [description], defaults to None
    :type contour_color_lims: [type], optional
    :param slice_keys: [description], defaults to None
    :type slice_keys: [type], optional
    :param slice_origins: [description], defaults to None
    :type slice_origins: [type], optional
    :param slice_normals: [description], defaults to None
    :type slice_normals: [type], optional
    :param slice_logs: [description], defaults to None
    :type slice_logs: [type], optional
    :param slice_cmaps: [description], defaults to None
    :type slice_cmaps: [type], optional
    :param slice_opacities: [description], defaults to None
    :type slice_opacities: [type], optional
    :param slice_color_lims: [description], defaults to None
    :type slice_color_lims: [type], optional
    :param is_orientation_axis: [description], defaults to False
    :type is_orientation_axis: bool, optional
    :param background_color: [description], defaults to None
    :type background_color: [type], optional
    :param camera_position: [description], defaults to None
    :type camera_position: [type], optional
    :param camera_focal_point: [description], defaults to None
    :type camera_focal_point: [type], optional
    :param camera_view_up: [description], defaults to None
    :type camera_view_up: [type], optional
    :param camera_dolly: [description], defaults to None
    :type camera_dolly: [type], optional
    :param resolution: [description], defaults to None
    :type resolution: [type], optional
    """

    print("EXECUTING PARAVIEW PYTHON SCRIPT")

    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    for file_no, file_path in enumerate(files):
        print("PROCESSING FILE NO %04d / %04d" % (file_no, len(files)-1))

        # create a new 'Xdmf3ReaderT'
        data_xdmf = Xdmf3ReaderT(registrationName='data.xdmf', FileName=[file_path])
        # data_xdmf.CellArrays = ['density', 'levelset', 'mach_number', 'mask_real', 'pressure', 'schlieren', 'temperature', 'velocity']

        # get active view
        renderView = GetActiveViewOrCreate('RenderView')
        # renderView = CreateRenderView()

        # CELL DATA TO POINT DATA
        PointData = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=data_xdmf)
        # PointData.CellDataArraytoprocess = ['density', 'levelset', 'mach_number', 'mask_real', 'pressure', 'schlieren', 'temperature', 'velocity']

        UpdatePipeline()
        bounds = PointData.GetDataInformation().GetBounds()
        positions_dict = get_positions_dict(bounds)

        # SLICES
        if slice_keys:

            for ii, (slice_ii_key, slice_ii_origin, slice_ii_normal) in enumerate(zip(slice_keys, slice_origins, slice_normals)):
                slice_ii_log = slice_logs[ii] if slice_logs else None
                slice_ii_cmap = slice_cmaps[ii] if slice_cmaps else None
                slice_ii_opacity = slice_opacities[ii] if slice_opacities else None
                slice_ii_color_lims = slice_color_lims[ii] if slice_color_lims else None
                slice_ii_translation = slice_translations[ii] if slice_translations else None
                
                slice_ii = Slice(registrationName='Slice' + str(ii), Input=PointData)
                slice_ii.SliceType = 'Plane'
                slice_ii.HyperTreeGridSlicer = 'Plane'
                slice_ii.SliceOffsetValues = [0.0]

                # init the 'Plane' selected for 'SliceType' / 'HyperTreeGridSlicer'
                slice_ii.SliceType.Origin = slice_ii_origin
                slice_ii.HyperTreeGridSlicer.Origin = slice_ii_origin

                # Properties modified on slice1.SliceType
                slice_ii.SliceType.Normal = slice_ii_normal

                sliceDisplay = Show(slice_ii, renderView, 'GeometryRepresentation')
                if slice_ii_opacity:
                    sliceDisplay.Opacity = slice_ii_opacity

                ColorBy(sliceDisplay, ('POINTS', slice_ii_key))

                colorMap = GetColorTransferFunction(slice_ii_key)
                opacityMap = GetOpacityTransferFunction(slice_ii_key)

                # # OPACITY MAPPING
                # colorMap.EnableOpacityMapping = 1

                # RESCALE
                if slice_ii_color_lims:
                    colorMap.RescaleTransferFunction(slice_ii_color_lims[0],slice_ii_color_lims[1])
                    opacityMap.RescaleTransferFunction(slice_ii_color_lims[0],slice_ii_color_lims[1])
                else:
                    colorMap.RescaleTransferFunctionToDataRange()
                    opacityMap.RescaleTransferFunctionToDataRange()

                # LOG SPACE
                if slice_ii_log:
                    colorMap.MapControlPointsToLogSpace()
                    colorMap.UseLogScale = 1
                    opacityMap.MapControlPointsToLogSpace()
                    opacityMap.UseLogScale = 1

                # CHOOSE COLOR MAP PRESET
                if slice_ii_cmap:
                    colorMap.ApplyPreset(get_paraview_preset(slice_ii_cmap), True)
                
                # TRANSFORMING
                if slice_ii_translation:
                    sliceDisplay.Position = slice_ii_translation
                    sliceDisplay.DataAxesGrid.Position = slice_ii_translation
                    sliceDisplay.PolarAxes.Translation = slice_ii_translation

        # CONTOURS
        if contour_color_keys:
            for ii, (contour_ii_key, contour_ii_value, contour_ii_color_key) in enumerate(zip(contour_keys, contour_values, contour_color_keys)):
                # SET CONTOUR ARGUMENTS
                contour_ii_opacity          = contour_opacities[ii] if contour_opacities else None
                contour_ii_specular         = contour_speculars[ii] if contour_speculars else None
                contour_ii_color_lims       = contour_color_lims[ii] if contour_color_lims else None
                contour_ii_cmap             = contour_cmaps[ii] if contour_cmaps else None
                contour_ii_opacity_mapping  = contour_opacity_mappings[ii] if contour_opacity_mappings else None

                # CONTOUR
                contour_ii = Contour(registrationName='Contour' + str(ii), Input=PointData)
                contour_ii.ContourBy = ['POINTS', contour_ii_key]
                contour_ii.Isosurfaces = [contour_ii_value]
                contour_ii.PointMergeMethod = 'Uniform Binning'

                # show data in view
                contourDisplay = Show(contour_ii, renderView, 'GeometryRepresentation')
                if contour_ii_opacity:
                    contourDisplay.Opacity = contour_ii_opacity

                if contour_ii_specular:
                    contourDisplay.Specular = contour_ii_specular
                
                ColorBy(contourDisplay, ('POINTS', contour_ii_color_key))

                # SHOW COLOR BAR ON/OFF
                # contourDisplay.SetScalarBarVisibility(renderView, True)
                # contourDisplay.SetScalarBarVisibility(renderView, False)

                colorMap   = GetColorTransferFunction(contour_ii_color_key)
                opacityMap = GetOpacityTransferFunction(contour_ii_color_key)

                if contour_ii_color_key is not None:
                    # RESCALE
                    if contour_ii_color_lims:
                        colorMap.RescaleTransferFunction(contour_ii_color_lims[0],contour_ii_color_lims[1])
                    else:
                        colorMap.RescaleTransferFunctionToDataRange()

                    # CHOOSE COLOR MAP PRESET
                    if contour_ii_cmap:
                        colorMap.ApplyPreset(get_paraview_preset(contour_ii_cmap), True)

                    # colorMap.EnableOpacityMapping = int(contour_ii_opacity_mapping)


        # SHOW ORIENTATION AXIS
        renderView.OrientationAxesVisibility = 1 if is_orientation_axis else 0

        # SET BACKGROUND COLOR
        if background_color:
            if background_color == "white":
                background_color = [1.0, 1.0, 1.0]
            colorPalette = GetSettingsProxy('ColorPalette')
            colorPalette.Background = background_color

        # # get layout
        # layout = GetLayout()
        # layout.SetSize(1576, 900)

        # SET CAMERA 
        camera = renderView.GetActiveCamera()

        if camera_position:
            if type(camera_position) == str:
                camera_position = positions_dict[camera_position]
            camera.SetPosition(camera_position)
        if camera_focal_point:
            if type(camera_focal_point) == str:
                camera_focal_point = positions_dict[camera_focal_point]
            camera.SetFocalPoint(camera_focal_point)
        if camera_view_up:
            camera.SetViewUp(camera_view_up)
        else:
            camera.SetViewUp((0.0, 0.0, 1.0))
        if camera_dolly:
            camera.Dolly(camera_dolly)
        else:
            camera.Dolly(1)

        camera.SetViewAngle(30)
        # camera.SetParallelProjection(False)
        # camera.SetParallelScale(0.24537112465547253)


        # ResetCamera()

        # SAVE SCREENSHOT
        if save_path:
            file_name = "image_%04d.png" % (file_no + naming_offset)
            file_save_path = os.path.join(save_path, file_name)
            SaveScreenshot(file_save_path, renderView, ImageResolution=resolution)

        # RECONNECT 
        Disconnect()
        Connect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=str, nargs="+")
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--naming_offset", type=int)

    parser.add_argument("--contour_keys", type=str, action="append")
    parser.add_argument("--contour_values", type=float, action="append")
    parser.add_argument("--contour_color_keys", type=str, action="append")
    parser.add_argument("--contour_cmaps", type=str, action="append")
    parser.add_argument("--contour_opacities", type=str, action="append")
    parser.add_argument("--contour_color_lims", type=str, nargs="+", action="append")
    parser.add_argument("--contour_speculars", type=str, action="append")
    parser.add_argument("--contour_opacity_mappings", type=str, action="append")

    parser.add_argument("--slice_keys", type=str, action="append")
    parser.add_argument("--slice_origins", type=float, nargs="+", action="append")
    parser.add_argument("--slice_normals", type=float, nargs="+", action="append")
    parser.add_argument("--slice_logs", type=str, action="append")
    parser.add_argument("--slice_cmaps", type=str, action="append")
    parser.add_argument("--slice_opacities", type=float, action="append")
    parser.add_argument("--slice_color_lims", type=float, nargs="+", action="append")
    parser.add_argument("--slice_translations", type=float, nargs="+", action="append")

    parser.add_argument("--camera_position", nargs="+")
    parser.add_argument("--camera_focal_point", nargs="+")
    parser.add_argument("--camera_view_up", type=float, nargs="+")
    parser.add_argument("--camera_view_angle", type=float, default=30)
    parser.add_argument("--camera_dolly", type=float, default=1.0)

    parser.add_argument("--is_orientation_axis", action="store_true")
    parser.add_argument("--background_color", nargs="+")
    parser.add_argument("--resolution", type=int, nargs="+")

    args = parser.parse_args()
    args_dict = vars(args)

    # CONTOURS
    if args_dict["contour_color_keys"]:
        for ii in range(len(args_dict["contour_color_keys"])):
            args_dict["contour_color_keys"][ii] = args_dict["contour_color_keys"][ii] if args_dict["contour_color_keys"][ii] != "None" else None
    if args_dict["contour_cmaps"]:
        for ii in range(len(args_dict["contour_cmaps"])):
            args_dict["contour_cmaps"][ii] = args_dict["contour_cmaps"][ii] if args_dict["contour_cmaps"][ii] != "None" else None
    if args_dict["contour_opacities"]:
        for ii in range(len(args_dict["contour_opacities"])):
            args_dict["contour_opacities"][ii] = float(args_dict["contour_opacities"][ii]) if args_dict["contour_opacities"][ii] != "None" else None
    if args_dict["contour_color_lims"]:
        for ii in range(len(args_dict["contour_color_lims"])):
            args_dict["contour_color_lims"][ii] = [float(args_dict["contour_color_lims"][ii][0]), float(args_dict["contour_color_lims"][ii][1])] if len(args_dict["contour_color_lims"][ii]) == 2 else None
    if args_dict["contour_speculars"]:
        for ii in range(len(args_dict["contour_speculars"])):
            args_dict["contour_speculars"][ii] = float(args_dict["contour_speculars"][ii]) if args_dict["contour_speculars"][ii] != "None" else None
    if args_dict["contour_opacity_mappings"]:
        for ii in range(len(args_dict["contour_opacity_mappings"])):
            args_dict["contour_opacity_mappings"][ii] = bool(args_dict["contour_opacity_mappings"][ii]) if args_dict["contour_opacity_mappings"][ii] != "None" else None

    # SLICES
    if args_dict["slice_logs"]:
        for ii in range(len(args_dict["slice_logs"])):
            args_dict["slice_logs"][ii] = True if args_dict["slice_logs"][ii] == "True" else False
    if args_dict["slice_cmaps"]:
        for ii in range(len(args_dict["slice_cmaps"])):
            args_dict["slice_cmaps"][ii] = args_dict["slice_cmaps"][ii] if args_dict["slice_cmaps"][ii] != "None" else None
    if args_dict["slice_opacities"]:
        for ii in range(len(args_dict["slice_opacities"])):
            args_dict["slice_opacities"][ii] = float(args_dict["slice_opacities"][ii]) if args_dict["slice_opacities"][ii] != "None" else None
    if args_dict["slice_color_lims"]:
        for ii in range(len(args_dict["slice_color_lims"])):
            args_dict["slice_color_lims"][ii] = [float(args_dict["slice_color_lims"][ii][0]), float(args_dict["slice_color_lims"][ii][1])] if len(args_dict["slice_color_lims"][ii]) == 2 else None

    # CAMERA
    if args_dict["camera_position"]: 
        args_dict["camera_position"]    = args_dict["camera_position"][0] if len(args_dict["camera_position"]) == 1 else [float(args_dict["camera_position"][ii]) for ii in range(3)]
    if args_dict["camera_focal_point"]:
        args_dict["camera_focal_point"] = args_dict["camera_focal_point"][0] if len(args_dict["camera_focal_point"]) == 1 else [float(args_dict["camera_focal_point"][ii]) for ii in range(3)]

    # MISC
    if args_dict["background_color"]:
        args_dict["background_color"] = args_dict["background_color"][0] if len(args_dict["background_color"]) == 1 else [float(args_dict["background_color"][ii]) for ii in range(3)]

    generate_paraview_pngs(**args_dict)