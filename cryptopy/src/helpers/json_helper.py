import json
import datetime
import os


class JsonHelper:
    @staticmethod
    def json_serial(obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return (
                obj.isoformat()
            )  # Convert to ISO format string (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
        raise TypeError(f"Type {type(obj)} not serializable")

    @staticmethod
    def save_to_json(data, filename):
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(filename, "w") as json_file:
            json.dump(data, json_file, indent=4, default=JsonHelper.json_serial)

    @staticmethod
    def read_from_json(filename):
        with open(filename, "r") as json_file:
            return json.load(json_file)
