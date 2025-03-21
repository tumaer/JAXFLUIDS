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
        "erdc_rainbow_bright": "erdc_rainbow_bright"
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
        files_points, 
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
        contour_color_bars=None,
        contour_color_bars_orientation=None,
        contour_color_bars_position=None,
        contour_color_bars_bold=None,
        contour_color_bars_font_size=None,
        contour_color_bars_bar_length=None,
        contour_color_bars_bar_thickness=None,
        slice_keys=None, 
        slice_origins=None, 
        slice_normals=None,
        slice_logs=None, 
        slice_cmaps=None, 
        slice_opacities=None, 
        slice_color_lims=None,
        slice_translations=None,
        slice_colorby_cells=None,
        slice_opacity_mappings=None,
        slice_threshold_key=None,
        slice_threshold_lims=None,
        slice_levelset_contour=None,
        slice_color_bars=None,
        slice_color_bars_orientation=None,
        slice_color_bars_position=None,
        slice_color_bars_bold=None,
        slice_color_bars_font_size=None,
        slice_color_bars_bar_length=None,
        slice_color_bars_bar_thickness=None,
        gradients_name=None,
        gradients_input_field=None,
        gradients_compute_divergence=None,
        gradients_compute_vorticity=None,
        gradients_compute_qcriterion=None,
        points_flag=None,
        points_size=None,
        points_opacity=None,
        points_color=None,
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


    for file_no, (file_path, file_points_path) in enumerate(zip(files, files_points)):
        print("PROCESSING FILE NO %04d / %04d" % (file_no, len(files)-1))

        # create a new 'Xdmf3ReaderT'
        data_xdmf = Xdmf3ReaderT(registrationName='data.xdmf', FileName=[file_path])
        # data_xdmf.CellArrays = ['density', 'levelset', 'mach_number', 'mask_real', 'pressure', 'schlieren', 'temperature', 'velocity']

        # get active view
        renderView = GetActiveViewOrCreate('RenderView')
        # renderView = CreateRenderView()

        # renderView.UseLight = 0
        
        # CELL DATA TO POINT DATA
        PointData = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=data_xdmf, PassCellData=1)
        # PointData.CellDataArraytoprocess = ['density', 'levelset', 'mach_number', 'mask_real', 'pressure', 'schlieren', 'temperature', 'velocity']

        UpdatePipeline()
        bounds = PointData.GetDataInformation().GetBounds()
        positions_dict = get_positions_dict(bounds)

        # COMPUTATION OF GRADIENTS
        gradients_dict = {}
        if gradients_name is not None:
            for ii, gradient_name in enumerate(gradients_name):
                gradient1 = Gradient(registrationName=gradient_name, Input=PointData)
                gradient1.ScalarArray = ['POINTS', gradients_input_field[ii]]
                if gradients_compute_divergence[ii]:
                    gradient1.ComputeDivergence = 1
                if gradients_compute_vorticity[ii]:
                    gradient1.ComputeVorticity = 1
                if gradients_compute_qcriterion[ii]:
                    gradient1.ComputeQCriterion = 1
            gradients_dict[gradient_name] = gradient1

        # SLICES
        if slice_keys:

            for ii, (slice_ii_key, slice_ii_origin, slice_ii_normal) in enumerate(zip(slice_keys, slice_origins, slice_normals)):
                slice_ii_log = slice_logs[ii] if slice_logs else None
                slice_ii_opacity_mappings = slice_opacity_mappings[ii] if slice_opacity_mappings else None
                slice_ii_cmap = slice_cmaps[ii] if slice_cmaps else None
                slice_ii_opacity = slice_opacities[ii] if slice_opacities else None
                slice_ii_color_lims = slice_color_lims[ii] if slice_color_lims else None
                slice_ii_translation = slice_translations[ii] if slice_translations else None
                slice_ii_colorby_cells = slice_colorby_cells[ii] if slice_colorby_cells else None
                slice_ii_threshold_key = slice_threshold_key[ii] if slice_threshold_key else None
                slice_ii_threshold_lims = slice_threshold_lims[ii] if slice_threshold_lims else None
                slice_ii_levelset_contour = slice_levelset_contour[ii] if slice_levelset_contour else None
                
                slice_ii = Slice(registrationName='Slice' + str(ii), Input=PointData)
                slice_ii.SliceType = 'Plane'
                slice_ii.HyperTreeGridSlicer = 'Plane'
                slice_ii.SliceOffsetValues = [0.0]

                colorby_cells = "CELLS" if slice_ii_colorby_cells else "POINTS"

                # init the 'Plane' selected for 'SliceType' / 'HyperTreeGridSlicer'
                slice_ii.SliceType.Origin = slice_ii_origin
                slice_ii.HyperTreeGridSlicer.Origin = slice_ii_origin

                # Properties modified on slice1.SliceType
                slice_ii.SliceType.Normal = slice_ii_normal

                sliceDisplay = Show(slice_ii, renderView, 'GeometryRepresentation')
                if isinstance(slice_ii_opacity, float):
                    sliceDisplay.Opacity = slice_ii_opacity
                ColorBy(sliceDisplay, (colorby_cells, slice_ii_key))

                # NOTE this disables lighting for slices
                sliceDisplay.Diffuse = 0
                sliceDisplay.Ambient = 1


                if slice_ii_key.endswith(("__X", "__Y", "__Z", "__Magnitude")):
                    slice_ii_key = slice_ii_key.split("__")
                    ColorBy(sliceDisplay, ('POINTS', *slice_ii_key))
                    colorMap = GetColorTransferFunction(slice_ii_key[0])
                    opacityMap = GetOpacityTransferFunction(slice_ii_key[0])
                    
                # TRANSFORM TRANSLATION
                if slice_ii_translation:
                    transform_ii = Transform(registrationName='Transform'+ str(ii), Input=slice_ii)
                    transform_ii.Transform.Translate = slice_ii_translation
                    transformDisplay = Show(transform_ii, renderView, 'GeometryRepresentation')
                    ColorBy(transformDisplay, (colorby_cells, slice_ii_key))
                    Hide(slice_ii, renderView)
                    colorMap = GetColorTransferFunction(slice_ii_key)
                    opacityMap = GetOpacityTransferFunction(slice_ii_key)
                
                # CONTOUR
                if slice_ii_levelset_contour:
                    contour = Contour(registrationName='Contour'+str(ii), Input=transform_ii)
                    contour.ContourBy = ['POINTS', 'levelset']
                    contour.Isosurfaces = [0.0]
                    contourDisplay = Show(contour, renderView, 'GeometryRepresentation')
                    contourDisplay.SetRepresentationType('Wireframe')
                    ColorBy(contourDisplay, None)
                    contourDisplay.AmbientColor = [0.0, 0.0, 0.0]
                    contourDisplay.DiffuseColor = [0.0, 0.0, 0.0]
                    contourDisplay.LineWidth = 10.0
                    contourDisplay.RenderLinesAsTubes = 1

                # THRESHOLD
                if slice_ii_threshold_key:
                    threshold_ii = Threshold(registrationName='Threshold'+ str(ii), Input=transform_ii)
                    threshold_ii.Scalars = ['CELLS', slice_ii_threshold_key]
                    threshold_ii.LowerThreshold = slice_ii_threshold_lims[0]
                    threshold_ii.UpperThreshold = slice_ii_threshold_lims[1]
                    thresholdDisplay = Show(threshold_ii, renderView, 'GeometryRepresentation')
                    ColorBy(thresholdDisplay, (colorby_cells, slice_ii_key))
                    Hide(transform_ii, renderView)
                    colorMap = GetColorTransferFunction(slice_ii_key)
                    opacityMap = GetOpacityTransferFunction(slice_ii_key)

                else:
                    ColorBy(sliceDisplay, ('POINTS', slice_ii_key))
                    colorMap = GetColorTransferFunction(slice_ii_key)
                    opacityMap = GetOpacityTransferFunction(slice_ii_key)

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
                
                if slice_ii_opacity_mappings:
                    colorMap.EnableOpacityMapping = 1

                # CHOOSE COLOR MAP PRESET
                if slice_ii_cmap:
                    colorMap.ApplyPreset(get_paraview_preset(slice_ii_cmap), True)

                # COLOR BAR
                if slice_color_bars is not None: 
                    if slice_color_bars[ii]:

                        sliceDisplay.SetScalarBarVisibility(renderView, True)
                        colorbar = GetScalarBar(colorMap, renderView)

                        if slice_color_bars_orientation[ii] in ("Horizontal", "Vertical"):
                            colorbar.AutoOrient = 0
                            colorbar.Orientation = slice_color_bars_orientation[ii]

                        if slice_color_bars_position[ii] != "None":
                            colorbar.WindowLocation = "Any Location"
                            colorbar.Position = slice_color_bars_position[ii]

                        if slice_color_bars_bold[ii]:
                            colorbar.TitleBold = 1
                            colorbar.LabelBold = 1

                        if slice_color_bars_font_size[ii] != "None":
                            colorbar.TitleFontSize = slice_color_bars_font_size[ii]
                            colorbar.LabelFontSize = slice_color_bars_font_size[ii]
                        
                        if slice_color_bars_bar_length[ii] != "None":
                            colorbar.ScalarBarLength = slice_color_bars_bar_length[ii]
                        
                        if slice_color_bars_bar_thickness[ii] != "None":
                            colorbar.ScalarBarThickness = slice_color_bars_bar_thickness[ii]

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
                if contour_ii_key in gradients_dict:
                    contour_ii = Contour(registrationName='Contour' + str(ii), Input=gradients_dict[contour_ii_key])
                else:
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

                if contour_ii_color_key is not None:
                    if contour_ii_color_key.endswith(("__X", "__Y", "__Z", "__Magnitude")):
                        contour_ii_color_key = contour_ii_color_key.split("__")
                    
                    # ColorBy(contourDisplay, ('POINTS', *contour_ii_color_key))
                    # colorMap = GetColorTransferFunction(contour_ii_color_key[0])
                    # opacityMap = GetOpacityTransferFunction(contour_ii_color_key[0])

                ColorBy(contourDisplay, ('POINTS', contour_ii_color_key))
                colorMap = GetColorTransferFunction(contour_ii_color_key)
                opacityMap = GetOpacityTransferFunction(contour_ii_color_key)


                # SOLID CONTOUR COLOR
                if contour_ii_key == "levelset":
                    # contourDisplay.AmbientColor = [0.0, 0.0, 1.0]
                    # contourDisplay.DiffuseColor = [0.0, 0.0, 1.0]
                    contourDisplay.DiffuseColor = [0.0, 0.3333333333333333, 1.0]
                    contourDisplay.AmbientColor = [0.0, 0.3333333333333333, 1.0]
                    contour_ii.ComputeNormals = 0
                    contourDisplay.SetRepresentationType('Surface With Edges')
                    contourDisplay.EdgeOpacity = 0.8
                    contourDisplay.EdgeColor = [0.0, 0.0, 0.0]
                else:
                    contour_ii.ComputeNormals = 0
                    contourDisplay.SetRepresentationType('Surface')
                    contourDisplay.EdgeOpacity = 0.0
                    contourDisplay.EdgeColor = [0.0, 0.0, 0.0]
                    contourDisplay.AmbientColor = [0.0, 0.0, 1.0]
                    contourDisplay.DiffuseColor = [0.0, 0.0, 1.0]

                # SHOW COLOR BAR ON/OFF
                # contourDisplay.SetScalarBarVisibility(renderView, True)
                # contourDisplay.SetScalarBarVisibility(renderView, False)

                # COLOR BAR
                if contour_color_bars is not None:
                    if contour_color_bars[ii]:

                        contourDisplay.SetScalarBarVisibility(renderView, True)
                        colorbar = GetScalarBar(colorMap, renderView)

                        if contour_color_bars_orientation[ii] in ("Horizontal", "Vertical"):
                            colorbar.AutoOrient = 0
                            colorbar.Orientation = contour_color_bars_orientation[ii]

                        if contour_color_bars_position[ii] != "None":
                            colorbar.WindowLocation = "Any Location"
                            colorbar.Position = contour_color_bars_position[ii]

                        if contour_color_bars_bold[ii]:
                            colorbar.TitleBold = 1
                            colorbar.LabelBold = 1

                        if contour_color_bars_font_size[ii] != "None":
                            colorbar.TitleFontSize = contour_color_bars_font_size[ii]
                            colorbar.LabelFontSize = contour_color_bars_font_size[ii]
                        
                        if contour_color_bars_bar_length[ii] != "None":
                            colorbar.ScalarBarLength = contour_color_bars_bar_length[ii]
                        
                        if contour_color_bars_bar_thickness[ii] != "None":
                            colorbar.ScalarBarThickness = contour_color_bars_bar_thickness[ii]

                if contour_ii_color_key is not None:
                    # RESCALE
                    if contour_ii_color_lims:
                        colorMap.RescaleTransferFunction(contour_ii_color_lims[0],contour_ii_color_lims[1])
                    else:
                        colorMap.RescaleTransferFunctionToDataRange()

                    # CHOOSE COLOR MAP PRESET
                    if contour_ii_cmap:
                        colorMap.ApplyPreset(get_paraview_preset(contour_ii_cmap), True)
                    
                    if contour_ii_opacity_mapping:
                        colorMap.EnableOpacityMapping = int(contour_ii_opacity_mapping)


        # SHOW ORIENTATION AXIS
        renderView.OrientationAxesVisibility = 1 if is_orientation_axis else 0

        # SET BACKGROUND COLOR
        if background_color:
            if isinstance(background_color, str):
                if background_color == "white":
                    background_color = (1.0, 1.0, 1.0)
                elif background_color == "black":
                    background_color = (0.0, 0.0, 0.0)
                else:
                    raise NotImplementedError
        
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

        camera.SetViewAngle(camera_view_angle)
        # camera.SetParallelProjection(False)
        # camera.SetParallelScale(0.24537112465547253)

        # # LIGHT
        # light = AddLight(view=renderView)
        # light.Coords = 'Ambient'
        # light.Enable = 1
        # light.Intensity = 0.15
        # light.Type = 'Positional' # Directional

        # ResetCamera()
        if points_flag:
            datatxt = PDALReader(registrationName='points', FileName=file_points_path)
            datatxtDisplay = Show(datatxt, renderView, 'GeometryRepresentation')
            datatxtDisplay.PointSize = points_size
            datatxtDisplay.RenderPointsAsSpheres = 1
            datatxtDisplay.AmbientColor = points_color
            datatxtDisplay.DiffuseColor = points_color
            datatxtDisplay.Opacity = points_opacity
            datatxtDisplay.Specular = 0.2

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
    parser.add_argument("--files_points", type=str, nargs="+")
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

    parser.add_argument("--contour_color_bars", type=int, action="append")
    parser.add_argument("--contour_color_bars_orientation", type=str, action="append")
    parser.add_argument("--contour_color_bars_position", type=float, nargs="+", action="append")
    parser.add_argument("--contour_color_bars_bold", type=int, action="append")
    parser.add_argument("--contour_color_bars_font_size", type=int, action="append")
    parser.add_argument("--contour_color_bars_bar_length", type=float, action="append")
    parser.add_argument("--contour_color_bars_bar_thickness", type=int, action="append")

    parser.add_argument("--slice_keys", type=str, action="append")
    parser.add_argument("--slice_origins", type=float, nargs="+", action="append")
    parser.add_argument("--slice_normals", type=float, nargs="+", action="append")
    parser.add_argument("--slice_logs", type=str, action="append")
    parser.add_argument("--slice_cmaps", type=str, action="append")
    parser.add_argument("--slice_opacities", type=float, action="append")
    parser.add_argument("--slice_color_lims", type=float, nargs="+", action="append")
    parser.add_argument("--slice_translations", type=float, nargs="+", action="append")
    parser.add_argument("--slice_colorby_cells", type=str, action="append")
    parser.add_argument("--slice_opacity_mappings", type=str, action="append")
    parser.add_argument("--slice_threshold_key", type=str, action="append")
    parser.add_argument("--slice_threshold_lims", type=float, nargs="+", action="append")
    parser.add_argument("--slice_levelset_contour", type=str, action="append")

    parser.add_argument("--points_flag", type=bool)
    parser.add_argument("--points_size", type=int)
    parser.add_argument("--points_opacity", type=float)
    parser.add_argument("--points_color", type=float, nargs="+")

    parser.add_argument("--slice_color_bars", type=int, action="append")
    parser.add_argument("--slice_color_bars_orientation", type=str, action="append")
    parser.add_argument("--slice_color_bars_position", type=float, nargs="+", action="append")
    parser.add_argument("--slice_color_bars_bold", type=int, action="append")
    parser.add_argument("--slice_color_bars_font_size", type=int, action="append")
    parser.add_argument("--slice_color_bars_bar_length", type=float, action="append")
    parser.add_argument("--slice_color_bars_bar_thickness", type=int, action="append")

    parser.add_argument("--gradients_name", type=str, action="append")
    parser.add_argument("--gradients_input_field", type=str, action="append")
    parser.add_argument("--gradients_compute_divergence", type=int, action="append")
    parser.add_argument("--gradients_compute_vorticity", type=int, action="append")
    parser.add_argument("--gradients_compute_qcriterion", type=int, action="append")

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
            args_dict["contour_opacity_mappings"][ii] = True if args_dict["contour_opacity_mappings"][ii] == "True" else False

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
    if args_dict["slice_colorby_cells"]:
        for ii in range(len(args_dict["slice_colorby_cells"])):
            args_dict["slice_colorby_cells"][ii] = True if args_dict["slice_colorby_cells"][ii] == "True" else False
    if args_dict["slice_opacity_mappings"]:
        for ii in range(len(args_dict["slice_opacity_mappings"])):
            args_dict["slice_opacity_mappings"][ii] = True if args_dict["slice_opacity_mappings"][ii] == "True" else False
    if args_dict["slice_threshold_key"]:
        for ii in range(len(args_dict["slice_threshold_key"])):
            args_dict["slice_threshold_key"][ii] = None if args_dict["slice_threshold_key"][ii] == "None" else args_dict["slice_threshold_key"][ii]
    if args_dict["slice_threshold_lims"]:
        for ii in range(len(args_dict["slice_threshold_lims"])):
            args_dict["slice_threshold_lims"][ii] = [float(args_dict["slice_threshold_lims"][ii][0]), float(args_dict["slice_threshold_lims"][ii][1])] if len(args_dict["slice_threshold_lims"][ii]) == 2 else None
    if args_dict["slice_levelset_contour"]:
        for ii in range(len(args_dict["slice_levelset_contour"])):
            args_dict["slice_levelset_contour"][ii] = True if args_dict["slice_levelset_contour"][ii] == "True" else False

    # GRADIENTS
    if args_dict["gradients_compute_divergence"]:
        args_dict["gradients_compute_divergence"] = [bool(val) for val in args_dict["gradients_compute_divergence"]]
    if args_dict["gradients_compute_vorticity"]:
        args_dict["gradients_compute_vorticity"] = [bool(val) for val in args_dict["gradients_compute_vorticity"]]
    if args_dict["gradients_compute_qcriterion"]:
        args_dict["gradients_compute_qcriterion"] = [bool(val) for val in args_dict["gradients_compute_qcriterion"]]

    if args_dict["slice_color_bars"]:
        args_dict["slice_color_bars"] = [bool(val) for val in args_dict["slice_color_bars"]]
    if args_dict["slice_color_bars_bold"]:
        args_dict["slice_color_bars_bold"] = [bool(val) for val in args_dict["slice_color_bars_bold"]]

    if args_dict["contour_color_bars"]:
        args_dict["contour_color_bars"] = [bool(val) for val in args_dict["contour_color_bars"]]
    if args_dict["contour_color_bars_bold"]:
        args_dict["contour_color_bars_bold"] = [bool(val) for val in args_dict["contour_color_bars_bold"]]

    # CAMERA
    if args_dict["camera_position"]: 
        args_dict["camera_position"] = args_dict["camera_position"][0] if len(args_dict["camera_position"]) == 1 else [float(args_dict["camera_position"][ii]) for ii in range(3)]
    if args_dict["camera_focal_point"]:
        args_dict["camera_focal_point"] = args_dict["camera_focal_point"][0] if len(args_dict["camera_focal_point"]) == 1 else [float(args_dict["camera_focal_point"][ii]) for ii in range(3)]

    # MISC
    if args_dict["background_color"]:
        args_dict["background_color"] = args_dict["background_color"][0] if len(args_dict["background_color"]) == 1 else [float(args_dict["background_color"][ii]) for ii in range(3)]
    
    generate_paraview_pngs(**args_dict)