#!/usr/bin/env python3
import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.b2.sport.sport_client import SportClient as SportClientB2
from unitree_sdk2py.go2.sport.sport_client import SportClient as SportClientGo2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal Unitree gait switch helper."
    )
    parser.add_argument(
        "--iface",
        default="enP8p1s0",
        help="Network interface connected to the robot.",
    )
    parser.add_argument(
        "--model",
        choices=("go2", "b2"),
        default="go2",
        help="Robot model sdk to use.",
    )
    parser.add_argument(
        "--mode",
        choices=("classic", "fast", "free"),
        default="classic",
        help="Target gait mode.",
    )
    parser.add_argument(
        "--disable",
        action="store_true",
        help="Disable the selected gait mode when supported.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ChannelFactoryInitialize(0, args.iface)
    client_cls = SportClientGo2 if args.model == "go2" else SportClientB2
    client = client_cls()
    client.SetTimeout(10.0)
    client.Init()
    time.sleep(1.0)

    enabled = not args.disable
    if args.mode == "classic":
        code = client.ClassicWalk(enabled)
    elif args.mode == "fast":
        code = client.FastWalk(enabled)
    else:
        if not enabled:
            raise ValueError("free mode does not support --disable")
        code = client.FreeWalk()

    print(f"model={args.model} mode={args.mode} enabled={enabled} return_code={code}")


if __name__ == "__main__":
    main()

