"""Run one validation case:  python src/main.py --case case1 [--elastic3d 1] [--save_data] [--plot] ...
Analyse the saved results separately:  python src/analysis.py --case case1"""
from args import parse_args
from simulation import Simulation as Sim

def main(argv=None):
    args = parse_args(argv)
    sim = Sim(
        case=args.case,
        elastic3d=args.elastic3d,
        save_video=args.save_video,
        save_data=args.save_data,
        override_simulation=args.override_simulation,
        override_xml=args.override_xml,
        xml=args.xml,
        debugger=args.debugger,
        viewer=not args.no_viewer,
        save_snapshots=args.save_snapshots,
    )

    sim.run()

    if (args.save_data or args.save_video or args.save_snapshots):
       # save if we said save the data/save the video 
       sim.save()

    if args.plot:
        import matplotlib.pyplot as plt
        from plotting import plot_run
        plot_run(*sim.results(), outdir=sim.outdir)
        plt.show()

    return sim


if __name__ == "__main__":
    main()
