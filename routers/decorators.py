import os
from flask import request, jsonify
from functools import wraps

API_TOKEN_SECRET = os.environ.get("API_TOKEN_SECRET", "e67d89305f2763bd4920bb0deda2e33467a9154522a99f5a6268e1a222ab9e75")

def require_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or token != f'Bearer {API_TOKEN_SECRET}':
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function