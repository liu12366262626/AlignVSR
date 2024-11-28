import os
import subprocess
import argparse

def extract_audio_ffmpeg(mp4_file, output_file, target_sample_rate=16000):
    """
    Extracts audio from an MP4 file and saves it as a mono WAV file with a 16kHz sample rate.

    Parameters:
    - mp4_file: Path to the input MP4 file.
    - output_file: Path to the output WAV file.
    - target_sample_rate: Desired sample rate for the audio (default is 16000).
    """
    command = [
        "ffmpeg", "-i", mp4_file,
        "-vn", "-acodec", "pcm_s16le", "-ar", str(target_sample_rate), "-ac", "1",  # Set sample rate to 16k, mono
        output_file
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def find_mp4_files(root_folder):
    """
    Recursively finds all MP4 files in the specified directory.

    Parameters:
    - root_folder: The folder to search for MP4 files.

    Returns:
    - A list of paths to all found MP4 files.
    """
    mp4_files = []
    
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.mp4'):
                mp4_files.append(os.path.join(dirpath, filename))
    
    return mp4_files

def extract_and_save_audio(mp4_file, root_folder, output_root, target_sample_rate=16000):
    """
    Extracts audio from an MP4 file and saves it to the target directory, preserving the original structure.

    Parameters:
    - mp4_file: Path to the input MP4 file.
    - root_folder: The root folder where the MP4 file is located.
    - output_root: The root folder where the extracted audio should be saved.
    - target_sample_rate: Desired sample rate for the extracted audio (default is 16000).
    """
    try:
        # Get the directory of the original MP4 file
        mp4_dir = os.path.dirname(mp4_file)
        
        # Build the relative path for the target audio directory
        relative_path = os.path.relpath(mp4_dir, start=root_folder)
        target_audio_dir = os.path.join(output_root, relative_path)
        
        # Create the target directory if it doesn't exist
        if not os.path.exists(target_audio_dir):
            os.makedirs(target_audio_dir)
        
        # Get the file name for the audio output
        filename = os.path.basename(mp4_file)
        audio_filename = f"{os.path.splitext(filename)[0]}.wav"
        audio_output_path = os.path.join(target_audio_dir, audio_filename)
        
        # Extract the audio and save it to the target folder
        extract_audio_ffmpeg(mp4_file, audio_output_path, target_sample_rate)
        
        print(f"Audio extracted and saved to {audio_output_path}")
        
    except Exception as e:
        print(f"Error extracting audio: {e}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Extract audio from MP4 files and save as WAV.")
    parser.add_argument('--root_folder', type=str, required=True, help="Path to the folder containing MP4 files.")
    parser.add_argument('--output_root', type=str, required=True, help="Path to the folder to save extracted audio.")
    
    args = parser.parse_args()
    
    # Find all MP4 files
    mp4_files = find_mp4_files(args.root_folder)
    
    # Extract and save audio for each MP4 file
    for mp4_file in mp4_files:
        extract_and_save_audio(mp4_file, args.root_folder, args.output_root)

if __name__ == "__main__":
    main()
