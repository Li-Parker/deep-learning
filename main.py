import argparse
parser = argparse.ArgumentParser(description="测试")

parser.add_argument("--in_channels", type=int, default=3, help="输入通道数")
parser.add_argument("--expect_result", type=float, default=3.0, help="期望结果")
args = parser.parse_args()
print(args.in_channels)
print(args.expect_result)