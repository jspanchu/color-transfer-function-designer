import vtk
from trame.app import get_server
from trame.ui.mui import SinglePageLayout
from trame.widgets import mui, react, vtklocal

server = get_server(client_type="react")
state, ctrl = server.state, server.controller
state.opacity = 0.5

src = vtk.vtkRTAnalyticSource()
src.SetWholeExtent(0, 40, 0, 40, 0, 40)
src.Update()
mapper = vtk.vtkGPUVolumeRayCastMapper()
mapper.SetInputConnection(src.GetOutputPort())
vol = vtk.vtkVolume()
vol.SetMapper(mapper)
otf = vtk.vtkPiecewiseFunction()
otf.AddPoint(0, 0)
otf.AddPoint(280, 1)
ctf = vtk.vtkColorTransferFunction()
ctf.AddRGBPoint(0, 0, 0, 1)
ctf.AddRGBPoint(280, 1, 0, 0)
vol.GetProperty().SetScalarOpacity(otf)
vol.GetProperty().SetColor(ctf)
ren = vtk.vtkRenderer()
rw = vtk.vtkRenderWindow()
rw.AddRenderer(ren)
rw.OffScreenRenderingOn()
ren.AddVolume(vol)
ren.ResetCamera()
rwi = vtk.vtkRenderWindowInteractor()
rwi.SetRenderWindow(rw)


@state.change("opacity")
def on_opacity(opacity, **_):
    otf.RemovePoint(280)
    otf.AddPoint(280, float(opacity))
    ctrl.view_update()


with SinglePageLayout(server) as layout:
    with layout.toolbar:
        mui.Typography("react smoke", variant="h6")
        mui.Slider(
            value=react.Bind("opacity"),
            min=0,
            max=1,
            step=0.05,
            on_change=react.Callback("opacity = Number($event.target.value)"),
            sx={"width": 200},
            marks=True,
            color="secondary",
        )
    with layout.content:
        view = vtklocal.LocalView(rw, throttle_rate=20)
        ctrl.view_update = view.update

if __name__ == "__main__":
    server.start()
