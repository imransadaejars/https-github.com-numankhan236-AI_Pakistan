##OPEN API STUFF
OPENAI_API_KEY = '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'



## FLASK STUFF
class Config(object):
    DEBUG = True
    TESTING = False

class DevelopmentConfig(Config):
    SECRET_KEY = "this-is-a-super-secret-key"


config = {
    'development': DevelopmentConfig,
    'testing': DevelopmentConfig,
    'production': DevelopmentConfig
}
