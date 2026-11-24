import argparse
## args: save video, save data, run certain case, with default settings for each case (src/settings.json),
## and optionally override them with command line arguments or a json file

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run MuJoCo flexcomp validation simulation with specified settings.")

    parser.add_argument('--case',
                        type=str,
                        choices=['case1', 'case2', 'case3'],
                        default='case1',
                        help='Simulation case to run (default: case1); settings per case in src/settings.json')

    parser.add_argument('--elastic3d',
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help='3D flex material: 0 = Saint Venant-Kirchhoff (MuJoCo default), 1 = stable neo-Hookean (stiffens). '
                             'SNH only settable through mjSpec')

    parser.add_argument('--save_video',
                        action='store_true',
                        help='Save a video (gif) of the simulation')

    parser.add_argument('--save_snapshots',
                        action='store_true',
                        help='Save png snapshots of the scene at first contact and at max indentation '
                             '(<case>_snapshot_first_contact.png / _max_indentation.png in the output folder)')

    parser.add_argument('--save_data',
                        action='store_true',
                        default=True,
                        help='Save simulation data (json) for the analysis')

    parser.add_argument('--xml',
                        type=str,
                        help='Path to an XML file to use instead of the one in the case settings')

    parser.add_argument('--override_simulation',
                        type=str,
                        help='Path to a JSON file with protocol settings that override the case settings, same structure '
                             'as one case in src/settings.json, e.g. {"protocol": {"speed": 0.002}} '
                             '(model settings go in --override_xml)')

    parser.add_argument('--override_xml',
                        type=str,
                        help='Path to a JSON file with model settings that override the case settings, e.g. '
                             '{"model": {"young": 110e3, "mesh_count": [21, 21, 8]}} (mesh_count: grid flexes only)')

    parser.add_argument('--debugger',
                        action='store_true',
                        help='Run the simulation in debugger mode and save extra data for debugging '
                             '(contacts per tool geom, max vertex speed, solver iterations)')

    parser.add_argument('--no_viewer',
                        action='store_true',
                        help='Run without the MuJoCo viewer window (faster)')

    parser.add_argument('--plot',
                        action='store_true',
                        help='After the simulation, plot force-displacement, power through the contact (nI) and '
                             'strain energy (saved as <case>_plot.png in the output folder)')

    args = parser.parse_args(argv)
    return args
