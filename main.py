from run_setup import run_once

if __name__ == "__main__":
    # defining the current trial of running
    trial_num = "250320"

    # running each condition for x times
    for run in range(2,3):
        run_once(trial_num, run, "txt", "harmony")
        run_once(trial_num, run, "txt", "disharmony")
        #run_once(trial_num, run, "aud", "harmony")
        #run_once(trial_num, run, "aud", "disharmony")