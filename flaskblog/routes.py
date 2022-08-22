import os
import secrets
from PIL import Image
from flask import render_template, url_for, flash, redirect, request, Response , jsonify
from flaskblog import app, db, bcrypt
from flaskblog.forms import RegistrationForm, LoginForm, UpdateAccountForm
from flaskblog.models import User, Post, Contact
from flask_login import login_user, current_user, logout_user, login_required
from flaskblog.ml_models import model
from flaskblog.camera import Video
from flaskblog import blog
from flaskblog.bot import get_response
from flaskblog.FaceEmotionCamera import VideoCamera
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from flaskblog.mask_camera import VideoCameraMask
import numpy as np

posts = [
    {
        'author': 'Corey Schafer',
        'title': 'Blog Post 1',
        'content': 'First post content',
        'date_posted': 'April 20, 2018'
    },
    {
        'author': 'Jane Doe',
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted': 'April 21, 2018'
    }
]


@app.route("/")
def main():
    return render_template('main.html', posts=posts)

@app.route("/home")
def home():
    return render_template('index.html', posts=posts)


@app.route("/about")
def about():
    return render_template('about.html', title='About')

@app.route("/contact", methods = ['POST', 'GET'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        subject = request.form['subject']
        message = request.form['message']
        new_contact = Contact(name = name , email = email , subject = subject , message = message)
        db.session.add(new_contact)
        db.session.commit()

    allContact = Contact.query.all()

    return render_template('contact.html', allContact = allContact)



@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
    else:
        flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('main'))


def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)

    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn


@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('account.html', title='Account',
                           image_file=image_file, form=form)






                           # Machine Learning Models


# spam messages detector
@app.route("/spam_detector",methods=['GET', 'POST'])
def spam_detector():
  if request.method == 'POST':
    message = request.form.get('message')
    output = model.predict([message])
    if output == [0]:
      result = "This Message is Not a SPAM Message."
    else:
      result = "This Message is a SPAM Message." 
    return render_template('spam_detector.html', result=result,message=message)      

  else:
    return render_template('spam_detector.html') 




# Face Detection

@app.route('/face_detector')
def face_detector():
    return render_template('face_detection.html')

def gen(camera):
    while True:
        frame=camera.get_frame()
        yield(b'--frame\r\n'
       b'Content-Type:  image/jpeg\r\n\r\n' + frame +
         b'\r\n\r\n')

@app.route('/video')

def video():
    return Response(gen(Video()),
    mimetype='multipart/x-mixed-replace; boundary=frame')



# AI Blog Writter

@app.route('/ai_writter', methods=["GET", "POST"])
def ai_writter():

    if request.method == 'POST':
        if 'form1' in request.form:
            prompt = request.form['blogTopic']
            blogT = blog.generateBlogTopics(prompt)
            blogTopicIdeas = blogT.replace('\n', '<br>')

        if 'form2' in request.form:
            prompt = request.form['blogSection']
            blogT = blog.generateBlogSections(prompt)
            blogSectionIdeas = blogT.replace('\n', '<br>')

        if 'form3' in request.form:
            prompt = request.form['blogExpander']
            blogT = blog.blogSectionExpander(prompt)
            blogExpanded = blogT.replace('\n', '<br>')


    return render_template('ai_writter.html', **locals())


# Website Chat Bot
@app.get('/home')
def base():
    return render_template('chatbot.html')

@app.post('/predict')
def pick_reply():
    msg = request.get_json().get('message')
    response = get_response(msg)
    message = {'answer': response}
    return jsonify(message)


# Facial Emotion Recognition

@app.route('/fer')
def fer():
    return render_template('FaceEmotionRecognition.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed1')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



# Mask Detection App

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = r"flaskblog\uploads"
STATIC_FOLDER = r"flaskblog\static"
CLASSIFY_FOLDER = "templates\classified_img"
CLASSIFIED_FOLDER = "classified_img"

print("[INFO] loading face detector model...")
prototxtPath = r'flaskblog\static\face_detector\deploy.prototxt'
weightsPath = r'flaskblog\static\face_detector\res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading face mask detector model...")
model = load_model('flaskblog\static\mask_detector.model')

# buat dewe
def classify(model, img_path,file):
    image = cv2.imread(img_path)
    orig = image.copy()
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))
    
    print("[INFO] computing face detections...")
    faceNet.setInput(blob)
    detections = faceNet.forward()

    label = ''
    prob = ''

    # keterangan kek di camera.py
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        if confidence > 0.5 :
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = model.predict(face)[0]

            label = "Prediction = Having Mask" if mask > withoutMask else "Prediction = Without Mask"
            color = (0, 255, 0) if label == "Prediction = Having Mask" else (0, 0, 255)

            prob = "{} {:.2f}%".format('Accuracy = ',max(mask, withoutMask) * 100)

            cv2.putText(image, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    save_processed_img = os.path.join(STATIC_FOLDER, file.filename)
    cv2.imwrite(save_processed_img,image)

    if label == '' or prob == '':
        label = "There is no human face in the picture"
        prob = 'Error'
    return label, prob


# home page

@app.route("/mask_detector")
def mask_home():
    return render_template("mask_home.html")

@app.route("/classificationbyimage")
def classificationbyimage():
    return render_template("mask_classificationbyimage.html")

@app.route("/classified", methods=["POST", "GET"])
def upload_file():

    if request.method == "GET":
        return render_template("mask_classificationbyimage.html")

    else:
        file = request.files["image"]

        # cek ada file gambar ga
        if file.filename == '':
            return redirect(url_for('classificationbyimage'))
        # print('file.filename == ',file.filename)

        # upload gambar
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print(upload_image_path)
        file.save(upload_image_path)

        label, prob = classify(model, upload_image_path,file)
        print('label == ',label,', prob == ',prob)
    return render_template(
        "mask_classify.html", image_name=file.filename, label=label, prob=prob
    )

@app.route('/mask_live_video')
def mask_live_video():
    return render_template('mask_live_video.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route("/video_feed")
def mask_video_feed():
    return Response(gen(VideoCameraMask()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
