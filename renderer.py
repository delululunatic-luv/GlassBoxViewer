import sys
from multiprocessing import Process, Queue
from vispy import app, scene
import torch
import numpy as np

def vispy_process(render_queue: Queue, visualizer):
    canvas = scene.SceneCanvas(title="GlassBoxViewer",
                               keys="interactive", bgcolor="black", size=(900, 700))

    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera()
    view.camera.clip_near = 0.001
    view.camera.clip_far = 1e11
    view.camera.translate_speed = 1000

    markers = scene.visuals.Markers()
    line_visual = scene.visuals.Line()
    markers.scaling = True
    view.add(markers)
    view.add(line_visual)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    initial_structure = False
    highest_z = []
    z_center = 0
    highest_y = []
    y_center = 0
    def update_linear(event):
        nonlocal initial_structure
        nonlocal highest_z
        nonlocal z_center
        nonlocal highest_y
        nonlocal y_center
        nonlocal visualizer

        if render_queue.empty():
            return
        render_info = render_queue.get_nowait()
        render_length = len(render_info)

        x_offset = 0
        x_offset_by = render_length
        y_offset = 10000 // next(iter(render_info.values()))["shape"].shape[1]
        z_offset = 10

        #layers = []
        lines = []

        for ren_name, ren_layer in render_info.items():
            if not initial_structure:
                view.camera.translate_speed = render_length
                coords = torch.nonzero(torch.ones_like(ren_layer['shape'], device=device), as_tuple=False)
                coords_np = coords.cpu().numpy()
                extra_dim = 0
                if coords_np.shape[1] > 3:
                    extra_dim = 1

                coords_np[:, 0 + extra_dim] += x_offset
                coords_np[:, 1 + extra_dim] *= y_offset
                if coords_np.shape[1] < 3:
                    coords_np = np.hstack([coords_np, np.zeros((coords_np.shape[0], 1))])
                coords_np[:, 2 + extra_dim] *= z_offset
                highest_z.append(np.max(coords_np[:, 2 + extra_dim]))
                #layers.append(coords_np)
                highest_y.append(np.max(coords_np[:, 1 + extra_dim]))

            lines_np = np.array(ren_layer['top_index_coords'], dtype=np.int32)
            if lines_np.shape[1] < 3:
                zeros = np.zeros((lines_np.shape[0], 1), dtype=lines_np.dtype)
                lines_np = np.hstack((zeros, lines_np))
            lines_np[:, 0] += x_offset
            lines_np[:, 1] *= y_offset
            if lines_np.shape[1] < 3:
                  lines_np = np.hstack([lines_np, np.zeros((lines_np.shape[0], 1))])
            lines_np[:, 2] *= z_offset

            lines.append(lines_np)

            x_offset += x_offset_by

        if not initial_structure:
            highest_z = np.array(highest_z, dtype=np.int64)
            if np.max(highest_z) == 0:
                highest_z[:] = 1
            mean_z = np.mean(highest_z)
            std_z = np.std(highest_z)
            if std_z == 0:
                std_z = 1.0
            filtered_z = highest_z[np.abs(highest_z - mean_z) < 2 * std_z]

            highest_y = np.array(highest_y, dtype=np.int64)
            if np.max(highest_y) == 0:
                highest_y[:] = 1
            mean_y = np.mean(highest_y)
            std_y = np.std(highest_y)
            if std_y == 0:
                std_y = 1.0
            filtered_y = highest_y[np.abs(highest_y - mean_y) < 2 * std_y]

            z_center = np.mean(filtered_z)
            y_center = np.mean(filtered_y)
            initial_structure = True

        line_visual.set_data(pos=lines, color='red', width=1, connect='strip')

        center = [x_offset//2, y_center//2, z_center]
        view.camera.center = center

        view.camera.distance = z_center * visualizer['camera_distance']

        canvas.update()




    def update_ring(event):
        view.camera.set_state(azimuth=90, elevation=0)

        nonlocal initial_structure
        nonlocal highest_z
        nonlocal z_center
        nonlocal highest_y
        nonlocal y_center

        def rotate_points_x(points, fraction=0.0):
            theta = fraction * 2 * np.pi
            c, s = np.cos(theta), np.sin(theta)

            rotated = points.copy()
            rotated[:, 1] = points[:, 1] * c - points[:, 2] * s
            rotated[:, 2] = points[:, 1] * s + points[:, 2] * c

            return rotated

        if render_queue.empty():
            return
        render_info = render_queue.get_nowait()
        render_length = len(render_info)

        x_offset = 0
        x_offset_by = render_length
        y_offset = 10000 // next(iter(render_info.values()))["shape"].shape[1]
        z_offset = 10

        #layers = []
        lines = []

        for i, (ren_name, ren_layer) in enumerate(render_info.items()):
            if not initial_structure:
                view.camera.translate_speed = render_length
                coords = torch.nonzero(torch.ones_like(ren_layer['shape'], device=device), as_tuple=False)
                coords_np = coords.cpu().numpy()

                extra_dim = 0
                if coords_np.shape[1] > 3:
                    extra_dim = 1

                coords_np[:, 0 + extra_dim] += x_offset
                coords_np[:, 1 + extra_dim] *= y_offset
                if coords_np.shape[1] < 3:
                    coords_np = np.hstack([coords_np, np.zeros((coords_np.shape[0], 1))])

                coords_np[:, 2 + extra_dim] *= z_offset
                highest_z.append(np.max(coords_np[:, 2 + extra_dim]))
                #layers.append(coords_np)
                highest_y.append(np.max(coords_np[:, 1 + extra_dim]))
            lines_np = np.array(ren_layer['top_index_coords'], dtype=np.int32)

            if lines_np.shape[1] < 3:
                zeros = np.zeros((lines_np.shape[0], 1), dtype=lines_np.dtype)
                lines_np = np.hstack((zeros, lines_np))

            lines_np[:, 0] += x_offset
            lines_np[:, 1] *= y_offset
            if lines_np.shape[1] < 3:
                  lines_np = np.hstack([lines_np, np.zeros((lines_np.shape[0], 1))])

            lines_np[:, 2] *= z_offset

            fraction = i / render_length
            line_rot = rotate_points_x(lines_np, fraction=fraction)

            lines.append(line_rot)

            x_offset += x_offset_by

        if not initial_structure:
            highest_z = np.array(highest_z, dtype=np.int64)
            if np.max(highest_z) == 0:
                highest_z[:] = 1
            mean_z = np.mean(highest_z)
            std_z = np.std(highest_z)
            if std_z == 0:
                std_z = 1.0
            filtered_z = highest_z[np.abs(highest_z - mean_z) < 2 * std_z]

            highest_y = np.array(highest_y, dtype=np.int64)
            if np.max(highest_y) == 0:
                highest_y[:] = 1
            mean_y = np.mean(highest_y)
            std_y = np.std(highest_y)
            if std_y == 0:
                std_y = 1.0
            filtered_y = highest_y[np.abs(highest_y - mean_y) < 2 * std_y]

            z_center = np.mean(filtered_z)
            y_center = np.mean(filtered_y)
            initial_structure = True

        line_visual.set_data(pos=lines, color='red', width=1, connect='strip')

        center = [x_offset // 2, 0, 0]
        view.camera.center = center
        view.camera.distance = z_center * visualizer['camera_distance']

        canvas.update()


    if visualizer['visual']  == 'linear':
        timer = app.Timer(interval=1/60, connect=update_linear, start=True)
    elif visualizer['visual'] == 'ring':
        timer = app.Timer(interval=1/60, connect=update_ring, start=True)
    canvas.show()
    app.run()


def start_render(render_queue, visualizer):
    p = Process(target=vispy_process, args=(render_queue, visualizer), daemon=True)
    p.start()
    return p