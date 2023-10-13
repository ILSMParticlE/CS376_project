import time
import subprocess

def parse_latest_submission(api_string):
    lines = api_string.split('\n')
    latest_line = list(filter(lambda x: x != '', lines[2].split(' ')))
    if latest_line[-3] == 'pending':
        return -1, -1
    public_score, private_score = float(latest_line[-2]), float(latest_line[-1])
    return public_score, private_score

def get_score(result_file='submission.csv'):
    public_score, private_score = -1, -1
    KAGGLE = 'kaggle'
    COMPETITIONS = 'competitions'
    SUBMIT = 'submit'
    SUBMISSIONS = 'submissions'
    COMPETITION_NAME = 'state-farm-distracted-driver-detection'
    FILENAME = result_file
    MESSAGE = 'api_submission'
    C_OPT = '-c'
    F_OPT = '-f'
    M_OPT = '-m'

    ''' Submit the submission file '''
    submit_command = [KAGGLE, COMPETITIONS, SUBMIT, C_OPT, COMPETITION_NAME, F_OPT, FILENAME, M_OPT, MESSAGE]
    result = subprocess.run(submit_command)
    print('')
    assert result.returncode == 0
    time.sleep(3)

    ''' Get public and private score '''
    submission_list_command = [KAGGLE, COMPETITIONS, SUBMISSIONS, COMPETITION_NAME]
    while public_score == private_score == -1:
        result = subprocess.run(submission_list_command, stdout=subprocess.PIPE, text=True)
        assert result.returncode == 0
        public_score, private_score = parse_latest_submission(result.stdout)
        time.sleep(0.5)

    return public_score, private_score

# if __name__ == '__main__':
#     public_score, private_score = get_score(result_file='result18_nopre.csv')
#     print(public_score, private_score)