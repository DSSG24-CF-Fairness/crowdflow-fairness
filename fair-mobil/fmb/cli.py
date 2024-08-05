import argparse
from .grav import grav_Model
from .rad import rad_Model

def main():
    parser = argparse.ArgumentParser(description='FairMobility CLI')
    
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.0.4')
    
    parser.add_argument('--grav', action='store_true', help='Run Gravity Model')
    parser.add_argument('--rad', action='store_true', help='Run Radiation Model')
    
    args = parser.parse_args()
    
    if args.grav:
        grav_Model()
    elif args.rad:
        rad_Model()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()