import os
import threading
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired
from PIL import Image
from torchvision import transforms
import io

# Import your existing AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(BASE_DIR, 'examples')

app = Flask(__name__)
app.config['SECRET_KEY']= 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png','jpg','jpeg'}
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'],exist_ok = True)

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha',default=1.0)
    submit = SubmitField('Transfer Style')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFERENCE_MAX_SIZE = 192

_models = None
_models_lock = threading.Lock()
_inference_lock = threading.Lock()

if device.type == "cpu":
    torch.set_num_threads(1)


def get_models() -> tuple[VGGEncoder, Decoder]:
    global _models
    if _models is None:
        with _models_lock:
            if _models is None:
                encoder = VGGEncoder(
                    os.path.join(BASE_DIR, 'utils', 'vgg_normalised.pth')
                ).to(device)
                decoder = Decoder().to(device)
                decoder_path = os.path.join(BASE_DIR, 'experiment', 'decoder_final.pth')
                decoder_weights = torch.load(
                    decoder_path, map_location='cpu', weights_only=True, mmap=True
                )
                decoder.load_state_dict({
                    key.replace("net.", "decoder.", 1): value
                    for key, value in decoder_weights.items()
                })
                encoder.eval()
                decoder.eval()
                _models = (encoder, decoder)
    models = _models
    if models is None:
        raise RuntimeError("Model initialization did not complete.")
    return models

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def style_transfer(content_image,style_image,encoder,decoder,alpha,device):
    content_image.thumbnail(
        (INFERENCE_MAX_SIZE, INFERENCE_MAX_SIZE), Image.Resampling.LANCZOS
    )
    style_image.thumbnail(
        (INFERENCE_MAX_SIZE, INFERENCE_MAX_SIZE), Image.Resampling.LANCZOS
    )
    to_tensor = transforms.ToTensor()
    content_image = to_tensor(content_image).unsqueeze(0).to(device)
    style_image = to_tensor(style_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats,style_feats)
        stylized_feats = alpha * stylized_feats + (1 - alpha)* content_feats

        stylized_image = decoder(stylized_feats)

    del content_image, style_image, content_feats, style_feats, stylized_feats
    return stylized_image

    
def save_image(image,path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0 , 1)
    image =transforms.ToPILImage()(image)
    image.save(path)


@app.route('/', methods = ['GET','POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        if form.content.data and form.content.data.filename :
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'],content_filename))
                form.content_path.data = content_filename
        else:
            content_filename = secure_filename(os.path.basename(form.content_path.data or ''))

        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'],style_filename))
                form.style_path.data = style_filename
        
        else:
            style_filename = secure_filename(os.path.basename(form.style_path.data or ''))

        if content_filename and style_filename:
            content_path =  os.path.join(app.config['UPLOAD_FOLDER'],content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'],style_filename)
            try:
              with Image.open(content_path) as source_image:
                  source_image.thumbnail(
                      (INFERENCE_MAX_SIZE, INFERENCE_MAX_SIZE),
                      Image.Resampling.LANCZOS,
                  )
                  content_image = source_image.convert('RGB')
              with Image.open(style_path) as source_image:
                  source_image.thumbnail(
                      (INFERENCE_MAX_SIZE, INFERENCE_MAX_SIZE),
                      Image.Resampling.LANCZOS,
                  )
                  style_image = source_image.convert('RGB')

              with _inference_lock:
                  encoder, decoder = get_models()
                  stylized_image = style_transfer(
                      content_image, style_image, encoder, decoder,
                      float(form.alpha.data), device
                  )

              result_filename = 'stylized' + content_filename
              result_path = os.path.join(app.config['UPLOAD_FOLDER'],result_filename)
              save_image(stylized_image,result_path)
              result_image =result_filename
            except Exception as e:
                error = str(e)            

    else:
        if not content_filename:
            error = 'Please upload content image'
        if not style_filename:
            error = 'Please upload style image '

    return render_template('index.html', form=form, result_image=result_image, content_image=content_filename,
                           style_image=style_filename, error=error)


@app.route('/features')
def features():
    return render_template('features.html')


@app.route('/pricing')
def pricing():
    return render_template('pricing.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')



@app.route('/uploads/<filename>')

def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/example/<path:filename>')

def send_example(filename):
    if filename == 'decoder_final_builtin_samples.png':
        return send_from_directory(os.path.join(BASE_DIR, 'experiment'), filename)

    category, separator, image_name = filename.partition('/')
    if separator and category == 'content':
        return send_from_directory(os.path.join(BASE_DIR, 'content_data'), image_name)
    if separator and category == 'style':
        return send_from_directory(os.path.join(BASE_DIR, 'style_data'), image_name)
    return send_from_directory(EXAMPLES_DIR, filename)

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost',5000,app,use_reloader=True,use_debugger=True)
