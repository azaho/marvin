import os
from pynwb import NWBHDF5IO


def test_load_nwb_data():
    # Path to the downloaded dataset
    dataset_path = "000070"

    # Example NWB file path
    nwb_file_path = os.path.join(
        dataset_path, "sub-Jenkins", "sub-Jenkins_ses-20090916_behavior+ecephys.nwb")

    # Verify file exists
    assert os.path.exists(
        nwb_file_path), f"NWB file not found at {nwb_file_path}"

    # Open the NWB file
    with NWBHDF5IO(nwb_file_path, 'r') as io:
        nwbfile = io.read()

        # Basic assertions to verify data loading
        assert nwbfile is not None, "Failed to load NWB file"
        assert hasattr(nwbfile, 'identifier'), "NWB file missing identifier"

        # Print some basic information about the file
        print(f"Successfully loaded NWB file: {nwbfile.identifier}")
        print(f"Session start time: {nwbfile.session_start_time}")


if __name__ == "__main__":
    test_load_nwb_data()
