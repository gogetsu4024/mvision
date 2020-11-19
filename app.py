from flask import Flask
from flask_restful import Resource, Api
import os
from routes import initialize_routes


UPLOAD_FOLDER = './'

app = Flask('__name__')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.debug = True
api = Api(app)


initialize_routes(api)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 6000))
    app.run(host='0.0.0.0', port=port)