import argparse
from segmentor import RoadSegmenter

def main():
    parser = argparse.ArgumentParser(description="Vehicle Tracking System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    segment_parser = subparsers.add_parser("segment", help="Run region/line selection")
    segment_parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    segment_parser.add_argument("--n", type=int, required=True, help="Number of iterations")

    tracking_parser = subparsers.add_parser("tracking", help="Run object tracking")
    tracking_parser.add_argument("--pipeline", type=str, default="yolov7", help="Path to the pipeline file (yolov11 or yolov7)")
    tracking_parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    tracking_parser.add_argument("--type", type=str, default="sort", help="Type of tracking algorithm to use (BoostTrack, BotSort, HybridSort, StrongSort, DeepOcSort, ByteTrack, OcSort)")

    args = parser.parse_args()

    if args.command == "segment":
        for _ in range(args.n):
            segmentor = RoadSegmenter(args.video, 0)
            segmentor.segment()
            print(f"Coordinate: {segmentor.coordinates}\n")

    elif args.command == "tracking":
        if args.pipeline not in ["yolov11", "yolov7"]:
            raise ValueError("Pipeline must be either 'yolov11' or 'yolov7'")
        
        if args.type not in ["sort", "BoostTrack", "BotSort", "HybridSort", "StrongSort", "DeepOcSort", "ByteTrack", "OcSort"]:
            raise ValueError("Tracking algorithm must be 'sort', 'BoostTrack', 'BotSort', 'HybridSort', 'StrongSort', 'DeepOcSort', 'ByteTrack', or 'OcSort'")
        
        if args.pipeline == "yolov11":
            from pipelineV11 import Pipeline
            pipline = Pipeline(args.config)
            pipline.run()
        elif args.pipeline == "yolov7":
            from pipelineV7 import PipelineV7
            pipline = PipelineV7(args.config, tracking_algorithm=args.type)
            pipline.run()
        else:
            raise ValueError("Unsupported pipeline type")

if __name__ == "__main__":
    main()