import multiprocessing

from color_transfer_function_designer.apps.medical.main import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main(exec_mode="desktop")
