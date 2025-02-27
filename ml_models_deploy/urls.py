from django.urls import path
from .views import classify_iris

urlpatterns = [
    path('ml_iris/knn/v01/predict/', classify_iris, name='ml_knn_iris_predict_v01'),
    path('ml_iris/lr/v01/predict/', classify_iris, name='ml_lr_iris_predict_v01'),
]
