from src import utils
import regex as re

def patten_match(result):  # because there is only 1 item in the list
    valid_results = []
    final_result = utils.text_from_json(result)
    print(final_result)
    # Expected pattern: [long alphanumeric]_[1-2 digits]_[2-5 letters]
    expected_pattern = r'^[0-9A-Z]{14,20}_\d{1,2}_[a-zA-Z]{1,5}$'
    for result in final_result:
        fixed_text = final_result[0]['fixed']
        if fixed_text and re.match(expected_pattern, fixed_text, re.IGNORECASE):
            valid_results.append(result)

    return valid_results