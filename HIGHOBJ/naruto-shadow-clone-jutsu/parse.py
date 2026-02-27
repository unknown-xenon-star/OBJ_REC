import argparse
def parse_args():
    parser = argparse.ArgumentParser(
        description="Naruto Shadow Clone in Python"
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)


if __name__ == "__main__":
    main()