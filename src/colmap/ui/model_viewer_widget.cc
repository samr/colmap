// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "colmap/ui/model_viewer_widget.h"

#include <algorithm>  //for replace
#include <fstream>
#include <iostream>
#include <list>
#include <math.h>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "colmap/ui/main_window.h"

#define SELECTION_BUFFER_IMAGE_IDX 0
#define SELECTION_BUFFER_POINT_IDX 1

const Eigen::Vector4f kSelectedPointColor(0.0f, 1.0f, 0.0f, 1.0f);

const Eigen::Vector4f kSelectedImagePlaneColor(1.0f, 0.0f, 1.0f, 0.6f);
const Eigen::Vector4f kSelectedImageFrameColor(0.8f, 0.0f, 0.8f, 1.0f);
const Eigen::Vector4f kSelectedFixedPathColor(0.0f, 0.8f, 0.8f, 1.0f);

const Eigen::Vector4f kMovieGrabberImagePlaneColor(0.0f, 1.0f, 1.0f, 0.6f);
const Eigen::Vector4f kMovieGrabberImageFrameColor(0.0f, 0.8f, 0.8f, 1.0f);

const Eigen::Vector4f kGridColor(0.2f, 0.2f, 0.2f, 0.6f);
const Eigen::Vector4f kXAxisColor(0.9f, 0.0f, 0.0f, 0.5f);
const Eigen::Vector4f kYAxisColor(0.0f, 0.9f, 0.0f, 0.5f);
const Eigen::Vector4f kZAxisColor(0.0f, 0.0f, 0.9f, 0.5f);

namespace colmap {
namespace {

std::pair<double, double> ComputeMeanAndStandardDeviation(
    const std::vector<double>& values) {
  const double length = static_cast<double>(values.size());

  double sum = 0;
  for (auto value : values) {
    sum += value;
  }
  const double mean = sum / length;

  double variance = 0;
  for (auto value : values) {
    variance += pow((value - mean), 2);
  }
  variance *= (1 / (length - 1));

  const double std_dev = std::sqrt(variance);
  return {mean, std_dev};
}

// Compute the mean over the trailing window for each next value in values.
std::vector<double> RollingMean(const std::vector<double>& values,
                                int window = 6) {
  std::vector<double> averaged_values;
  averaged_values.reserve(values.size());
  for (int i = 0; i < static_cast<int>(values.size()); ++i) {
    double total = 0.0;
    int num_values = 0;
    for (int j = 0; j < window; ++j) {
      const int k = i - j;
      if (k >= 0) {
        total += values[k];
        ++num_values;
      }
    }
    averaged_values.push_back(total / static_cast<double>(num_values));
  }
  return averaged_values;
}

// Only return the indices for values that are in the lower 3 quartiles
// of the normal distribution.
std::vector<size_t> IdentifyInlierIndices(const std::vector<double>& values,
                                          double mean, double std_dev) {
  const double upper_limit = mean + 3 * std_dev;
  std::vector<size_t> inlier_indices;
  for (int i = 0; i < values.size(); i++) {
    if (values[i] <= upper_limit) {
      inlier_indices.push_back(i);
    } else {
      std::cout << StringPrintf("%zu: removed outlier %.2f\n", i, values[i]);
    }
  }
  return inlier_indices;
}

// Returns the indices of the inliers, where outliers are determined by taking
// the difference between the measured values and the rolling mean of the
// values, looking at the normal distribution and removing any that are in the
// upper quartile.
std::vector<size_t> RollingOutlierFilter(const std::vector<double>& sensor,
                                         int window = 6) {
  const auto vec_mean = RollingMean(sensor, window);

  // Calculate the difference between the real values and the rolling mean.
  std::vector<double> vec_mean_diff;
  vec_mean_diff.reserve(vec_mean.size());
  for (int i = 0; i < sensor.size(); i++) {
    vec_mean_diff.push_back(sensor[i] - vec_mean[i]);
  }

  // Compute the standard deviation on the difference and use it to select
  // inliers.
  const auto mean_and_std = ComputeMeanAndStandardDeviation(vec_mean_diff);
  const auto mean = mean_and_std.first;
  const auto std_dev = mean_and_std.second;
  // std::cout << StringPrintf("mean=%f, std_dev=%f\n", mean, std_dev);
  return IdentifyInlierIndices(vec_mean_diff, mean, std_dev);
}

// Generate unique index from RGB color in the range [0, 256^3].
inline size_t RGBToIndex(const uint8_t r, const uint8_t g, const uint8_t b) {
  return static_cast<size_t>(r) + static_cast<size_t>(g) * 256 +
         static_cast<size_t>(b) * 65536;
}

// Derive color from unique index, generated by `RGBToIndex`.
inline Eigen::Vector4f IndexToRGB(const size_t index) {
  Eigen::Vector4f color;
  color(0) = ((index & 0x000000FF) >> 0) / 255.0f;
  color(1) = ((index & 0x0000FF00) >> 8) / 255.0f;
  color(2) = ((index & 0x00FF0000) >> 16) / 255.0f;
  color(3) = 1.0f;
  return color;
}

void BuildImageModel(const Image& image,
                     const Camera& camera,
                     const float image_size,
                     const Eigen::Vector4f& plane_color,
                     const Eigen::Vector4f& frame_color,
                     std::vector<TrianglePainter::Data>* triangle_data,
                     std::vector<LinePainter::Data>* line_data) {
  // Generate camera dimensions in OpenGL (world) coordinate space.
  const float kBaseCameraWidth = 1024.0f;
  const float image_width = image_size * camera.Width() / kBaseCameraWidth;
  const float image_height = image_width * static_cast<float>(camera.Height()) /
                             static_cast<float>(camera.Width());
  const float image_extent = std::max(image_width, image_height);
  const float camera_extent = std::max(camera.Width(), camera.Height());
  const float camera_extent_normalized =
      static_cast<float>(camera.CamFromImgThreshold(camera_extent));
  const float focal_length = 2.0f * image_extent / camera_extent_normalized;

  const Eigen::Matrix<float, 3, 4> inv_proj_matrix =
      Inverse(image.CamFromWorld()).ToMatrix().cast<float>();

  // Projection center, top-left, top-right, bottom-right, bottom-left corners.

  const Eigen::Vector3f pc = inv_proj_matrix.rightCols<1>();
  const Eigen::Vector3f tl =
      inv_proj_matrix *
      Eigen::Vector4f(-image_width, image_height, focal_length, 1);
  const Eigen::Vector3f tr =
      inv_proj_matrix *
      Eigen::Vector4f(image_width, image_height, focal_length, 1);
  const Eigen::Vector3f br =
      inv_proj_matrix *
      Eigen::Vector4f(image_width, -image_height, focal_length, 1);
  const Eigen::Vector3f bl =
      inv_proj_matrix *
      Eigen::Vector4f(-image_width, -image_height, focal_length, 1);

  // Image plane as two triangles.
  if (triangle_data != nullptr) {
    triangle_data->emplace_back(PointPainter::Data(tl(0),
                                                   tl(1),
                                                   tl(2),
                                                   plane_color(0),
                                                   plane_color(1),
                                                   plane_color(2),
                                                   plane_color(3)),
                                PointPainter::Data(tr(0),
                                                   tr(1),
                                                   tr(2),
                                                   plane_color(0),
                                                   plane_color(1),
                                                   plane_color(2),
                                                   plane_color(3)),
                                PointPainter::Data(bl(0),
                                                   bl(1),
                                                   bl(2),
                                                   plane_color(0),
                                                   plane_color(1),
                                                   plane_color(2),
                                                   plane_color(3)));

    triangle_data->emplace_back(PointPainter::Data(bl(0),
                                                   bl(1),
                                                   bl(2),
                                                   plane_color(0),
                                                   plane_color(1),
                                                   plane_color(2),
                                                   plane_color(3)),
                                PointPainter::Data(tr(0),
                                                   tr(1),
                                                   tr(2),
                                                   plane_color(0),
                                                   plane_color(1),
                                                   plane_color(2),
                                                   plane_color(3)),
                                PointPainter::Data(br(0),
                                                   br(1),
                                                   br(2),
                                                   plane_color(0),
                                                   plane_color(1),
                                                   plane_color(2),
                                                   plane_color(3)));
  }

  if (line_data != nullptr) {
    // Frame around image plane and connecting lines to projection center.

    line_data->emplace_back(PointPainter::Data(pc(0),
                                               pc(1),
                                               pc(2),
                                               frame_color(0),
                                               frame_color(1),
                                               frame_color(2),
                                               frame_color(3)),
                            PointPainter::Data(tl(0),
                                               tl(1),
                                               tl(2),
                                               frame_color(0),
                                               frame_color(1),
                                               frame_color(2),
                                               frame_color(3)));

    line_data->emplace_back(PointPainter::Data(pc(0),
                                               pc(1),
                                               pc(2),
                                               frame_color(0),
                                               frame_color(1),
                                               frame_color(2),
                                               frame_color(3)),
                            PointPainter::Data(tr(0),
                                               tr(1),
                                               tr(2),
                                               frame_color(0),
                                               frame_color(1),
                                               frame_color(2),
                                               frame_color(3)));

    line_data->emplace_back(PointPainter::Data(pc(0),
                                               pc(1),
                                               pc(2),
                                               frame_color(0),
                                               frame_color(1),
                                               frame_color(2),
                                               frame_color(3)),
                            PointPainter::Data(br(0),
                                               br(1),
                                               br(2),
                                               frame_color(0),
                                               frame_color(1),
                                               frame_color(2),
                                               frame_color(3)));

    line_data->emplace_back(PointPainter::Data(pc(0),
                                               pc(1),
                                               pc(2),
                                               frame_color(0),
                                               frame_color(1),
                                               frame_color(2),
                                               frame_color(3)),
                            PointPainter::Data(bl(0),
                                               bl(1),
                                               bl(2),
                                               frame_color(0),
                                               frame_color(1),
                                               frame_color(2),
                                               frame_color(3)));

    line_data->emplace_back(PointPainter::Data(tl(0),
                                               tl(1),
                                               tl(2),
                                               frame_color(0),
                                               frame_color(1),
                                               frame_color(2),
                                               frame_color(3)),
                            PointPainter::Data(tr(0),
                                               tr(1),
                                               tr(2),
                                               frame_color(0),
                                               frame_color(1),
                                               frame_color(2),
                                               frame_color(3)));

    line_data->emplace_back(PointPainter::Data(tr(0),
                                               tr(1),
                                               tr(2),
                                               frame_color(0),
                                               frame_color(1),
                                               frame_color(2),
                                               frame_color(3)),
                            PointPainter::Data(br(0),
                                               br(1),
                                               br(2),
                                               frame_color(0),
                                               frame_color(1),
                                               frame_color(2),
                                               frame_color(3)));

    line_data->emplace_back(PointPainter::Data(br(0),
                                               br(1),
                                               br(2),
                                               frame_color(0),
                                               frame_color(1),
                                               frame_color(2),
                                               frame_color(3)),
                            PointPainter::Data(bl(0),
                                               bl(1),
                                               bl(2),
                                               frame_color(0),
                                               frame_color(1),
                                               frame_color(2),
                                               frame_color(3)));

    line_data->emplace_back(PointPainter::Data(bl(0),
                                               bl(1),
                                               bl(2),
                                               frame_color(0),
                                               frame_color(1),
                                               frame_color(2),
                                               frame_color(3)),
                            PointPainter::Data(tl(0),
                                               tl(1),
                                               tl(2),
                                               frame_color(0),
                                               frame_color(1),
                                               frame_color(2),
                                               frame_color(3)));
  }
}

}  // namespace

ModelViewerWidget::ModelViewerWidget(QWidget* parent, OptionManager* options)
    : QOpenGLWidget(parent),
      options_(options),
      point_viewer_widget_(new PointViewerWidget(parent, this, options)),
      image_viewer_widget_(
          new DatabaseImageViewerWidget(parent, this, options)),
      movie_grabber_widget_(new MovieGrabberWidget(parent, this)),
      mouse_is_pressed_(false),
      focus_distance_(kInitFocusDistance),
      selected_image_id_(kInvalidImageId),
      selected_point3D_id_(kInvalidPoint3DId),
      coordinate_grid_enabled_(true),
      near_plane_(kInitNearPlane) {
  background_color_[0] = 1.0f;
  background_color_[1] = 1.0f;
  background_color_[2] = 1.0f;

  QSurfaceFormat format;
  format.setDepthBufferSize(24);
  format.setMajorVersion(3);
  format.setMinorVersion(2);
  format.setSamples(4);
  format.setProfile(QSurfaceFormat::CoreProfile);
#ifdef DEBUG
  format.setOption(QSurfaceFormat::DebugContext);
#endif
  setFormat(format);
  QSurfaceFormat::setDefaultFormat(format);

  SetPointColormap(new PointColormapPhotometric());
  SetImageColormap(new ImageColormapUniform());

  image_size_ = static_cast<float>(devicePixelRatio() * image_size_);
  point_size_ = static_cast<float>(devicePixelRatio() * point_size_);
}

void ModelViewerWidget::initializeGL() {
  initializeOpenGLFunctions();
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
  SetupPainters();
  SetupView();
}

void ModelViewerWidget::paintGL() {
  glClearColor(
      background_color_[0], background_color_[1], background_color_[2], 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  const QMatrix4x4 pmv_matrix = projection_matrix_ * model_view_matrix_;

  // Model view matrix for center of view
  QMatrix4x4 model_view_center_matrix = model_view_matrix_;
  const Eigen::Vector4f rot_center =
      QMatrixToEigen(model_view_matrix_).inverse() *
      Eigen::Vector4f(0, 0, -focus_distance_, 1);
  model_view_center_matrix.translate(
      rot_center(0), rot_center(1), rot_center(2));

  // Coordinate system
  if (coordinate_grid_enabled_) {
    const QMatrix4x4 pmvc_matrix =
        projection_matrix_ * model_view_center_matrix;
    coordinate_axes_painter_.Render(pmv_matrix, width(), height(), 2);
    coordinate_grid_painter_.Render(pmvc_matrix, width(), height(), 1);
  }

  // Points
  point_painter_.Render(pmv_matrix, point_size_);
  point_connection_painter_.Render(pmv_matrix, width(), height(), 1);

  // Images
  image_line_painter_.Render(pmv_matrix, width(), height(), 1);
  image_triangle_painter_.Render(pmv_matrix);
  image_connection_painter_.Render(pmv_matrix, width(), height(), 1);
  image_time_line_painter_.Render(pmv_matrix, width(), height(), 1);

  // Movie grabber cameras
  movie_grabber_path_painter_.Render(pmv_matrix, width(), height(), 1.5);
  movie_grabber_line_painter_.Render(pmv_matrix, width(), height(), 1);
  movie_grabber_triangle_painter_.Render(pmv_matrix);
}

void ModelViewerWidget::resizeGL(int width, int height) {
  glViewport(0, 0, width, height);
  ComposeProjectionMatrix();
  UploadCoordinateGridData();
}

void ModelViewerWidget::ReloadReconstruction() {
  if (reconstruction == nullptr) {
    return;
  }

  cameras = reconstruction->Cameras();
  points3D = reconstruction->Points3D();
  reg_image_ids = reconstruction->RegImageIds();

  images.clear();
  for (const image_t image_id : reg_image_ids) {
    images[image_id] = reconstruction->Image(image_id);
  }

  // Parse images with numbers in their names, e.g. out00001.png
  const size_t num_images = reg_image_ids.size();
  if (num_images > 0) {
    name_nums_to_image_indices.clear();
    name_nums_to_image_indices.resize(num_images);
    QRegExp regex("(\\d+)");  // Capture the first group of digits
    for (size_t i = 0; i < num_images; ++i) {
      const Image& image = images[reg_image_ids[i]];
      std::ignore = regex.indexIn(QString(image.Name().data()));
      QStringList captures = regex.capturedTexts();
      bool success = false;
      size_t image_num;
      if (captures.count() > 0) {
        success = true;
        image_num = static_cast<size_t>(captures[1].toInt(&success));
      }
      if (!success) {
        image_num = std::numeric_limits<size_t>::max() - num_images + i;
      }
      name_nums_to_image_indices[i] = {image_num, i};
    }
    std::sort(name_nums_to_image_indices.begin(),
              name_nums_to_image_indices.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
  }

  statusbar_status_label->setText(
      QString().asprintf("%d Images - %d Points",
                         static_cast<int>(reg_image_ids.size()),
                         static_cast<int>(points3D.size())));

  Upload();
}

void ModelViewerWidget::ClearReconstruction() {
  cameras.clear();
  images.clear();
  points3D.clear();
  reg_image_ids.clear();
  reconstruction = nullptr;
  Upload();
}

int ModelViewerWidget::GetProjectionType() const {
  return options_->render->projection_type;
}

void ModelViewerWidget::SetPointColormap(PointColormapBase* colormap) {
  point_colormap_.reset(colormap);
}

void ModelViewerWidget::SetImageColormap(ImageColormapBase* colormap) {
  image_colormap_.reset(colormap);
}

void ModelViewerWidget::UpdateMovieGrabber() {
  UploadMovieGrabberData();
  update();
}

void ModelViewerWidget::EnableCoordinateGrid() {
  coordinate_grid_enabled_ = true;
  update();
}

void ModelViewerWidget::DisableCoordinateGrid() {
  coordinate_grid_enabled_ = false;
  update();
}

void ModelViewerWidget::ChangeFocusDistance(const float delta) {
  if (delta == 0.0f) {
    return;
  }
  const float prev_focus_distance = focus_distance_;
  float diff = delta * ZoomScale() * kFocusSpeed;
  focus_distance_ -= diff;
  if (focus_distance_ < kMinFocusDistance) {
    focus_distance_ = kMinFocusDistance;
    diff = prev_focus_distance - focus_distance_;
  } else if (focus_distance_ > kMaxFocusDistance) {
    focus_distance_ = kMaxFocusDistance;
    diff = prev_focus_distance - focus_distance_;
  }
  const Eigen::Matrix4f vm_mat = QMatrixToEigen(model_view_matrix_).inverse();
  const Eigen::Vector3f tvec(0, 0, diff);
  const Eigen::Vector3f tvec_rot = vm_mat.block<3, 3>(0, 0) * tvec;
  model_view_matrix_.translate(tvec_rot(0), tvec_rot(1), tvec_rot(2));
  ComposeProjectionMatrix();
  UploadCoordinateGridData();
  update();
}

void ModelViewerWidget::ChangeNearPlane(const float delta) {
  if (delta == 0.0f) {
    return;
  }
  near_plane_ *= (1.0f + delta / 100.0f * kNearPlaneScaleSpeed);
  near_plane_ = std::max(kMinNearPlane, std::min(kMaxNearPlane, near_plane_));
  ComposeProjectionMatrix();
  UploadCoordinateGridData();
  update();
}

void ModelViewerWidget::ChangePointSize(const float delta) {
  if (delta == 0.0f) {
    return;
  }
  point_size_ *= (1.0f + delta / 100.0f * kPointScaleSpeed);
  point_size_ = std::max(kMinPointSize, std::min(kMaxPointSize, point_size_));
  update();
}

void ModelViewerWidget::RotateView(const float x,
                                   const float y,
                                   const float prev_x,
                                   const float prev_y) {
  if (x - prev_x == 0 && y - prev_y == 0) {
    return;
  }

  // Rotation according to the Arcball method "ARCBALL: A User Interface for
  // Specifying Three-Dimensional Orientation Using a Mouse", Ken Shoemake,
  // University of Pennsylvania, 1992.

  // Determine Arcball vector on unit sphere.
  const Eigen::Vector3f u = PositionToArcballVector(x, y);
  const Eigen::Vector3f v = PositionToArcballVector(prev_x, prev_y);

  // Angle between vectors.
  const float angle = 2.0f * std::acos(std::min(1.0f, u.dot(v)));

  const float kMinAngle = 1e-3f;
  if (angle > kMinAngle) {
    const Eigen::Matrix4f vm_mat = QMatrixToEigen(model_view_matrix_).inverse();

    // Rotation axis.
    Eigen::Vector3f axis = vm_mat.block<3, 3>(0, 0) * v.cross(u);
    axis = axis.normalized();
    // Center of rotation is current focus.
    const Eigen::Vector4f rot_center =
        vm_mat * Eigen::Vector4f(0, 0, -focus_distance_, 1);
    // First shift to rotation center, then rotate and shift back.
    model_view_matrix_.translate(rot_center(0), rot_center(1), rot_center(2));
    model_view_matrix_.rotate(RadToDeg(angle), axis(0), axis(1), axis(2));
    model_view_matrix_.translate(
        -rot_center(0), -rot_center(1), -rot_center(2));
    update();
  }
}

void ModelViewerWidget::TranslateView(const float x,
                                      const float y,
                                      const float prev_x,
                                      const float prev_y) {
  if (x - prev_x == 0 && y - prev_y == 0) {
    return;
  }

  Eigen::Vector3f tvec(x - prev_x, prev_y - y, 0.0f);

  if (options_->render->projection_type ==
      RenderOptions::ProjectionType::PERSPECTIVE) {
    tvec *= ZoomScale();
  } else if (options_->render->projection_type ==
             RenderOptions::ProjectionType::ORTHOGRAPHIC) {
    tvec *= 2.0f * OrthographicWindowExtent() / height();
  }

  const Eigen::Matrix4f vm_mat = QMatrixToEigen(model_view_matrix_).inverse();

  const Eigen::Vector3f tvec_rot = vm_mat.block<3, 3>(0, 0) * tvec;
  model_view_matrix_.translate(tvec_rot(0), tvec_rot(1), tvec_rot(2));

  update();
}

void ModelViewerWidget::ChangeCameraSize(const float delta) {
  if (delta == 0.0f) {
    return;
  }
  image_size_ *= (1.0f + delta / 100.0f * kImageScaleSpeed);
  image_size_ = std::max(kMinImageSize, std::min(kMaxImageSize, image_size_));
  UploadImageData();
  UploadMovieGrabberData();
  UploadImageTimeData();
  update();
}

void ModelViewerWidget::ResetView() {
  SetupView();
  Upload();
}

void ModelViewerWidget::SelectImage(size_t image_num, size_t image_num_type) {
  switch (image_num_type) {
    case 0:  // Select using the number in the filename.
      for (const auto& num_and_index : name_nums_to_image_indices) {
        if (num_and_index.first == image_num) {
          selected_image_id_ =
              static_cast<image_t>(reg_image_ids[num_and_index.second]);
          break;
        }
      }
      break;
    case 1:  // Select using the image_id.
      for (const auto& image_id : reg_image_ids) {
        if (image_id == image_num) {
          selected_image_id_ = static_cast<image_t>(image_num);
          break;
        }
      }
      break;
  }
  selected_point3D_id_ = kInvalidPoint3DId;
  ShowImageInfo(selected_image_id_);
  UploadImageData();
  update();
}

QMatrix4x4 ModelViewerWidget::ModelViewMatrix() const {
  return model_view_matrix_;
}

void ModelViewerWidget::SetModelViewMatrix(const QMatrix4x4& matrix) {
  model_view_matrix_ = matrix;
  update();
}

void ModelViewerWidget::SelectObject(const int x, const int y) {
  makeCurrent();

  // Ensure that anti-aliasing does not change the colors of objects.
  glDisable(GL_MULTISAMPLE);

  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Upload data in selection mode (one color per object).
  UploadImageData(true);
  UploadPointData(true);
  UploadImageTimeData(true);

  // Render in selection mode, with larger points to improve selection accuracy.
  const QMatrix4x4 pmv_matrix = projection_matrix_ * model_view_matrix_;
  image_triangle_painter_.Render(pmv_matrix);
  point_painter_.Render(pmv_matrix, 2 * point_size_);

  const int scaled_x = devicePixelRatio() * x;
  const int scaled_y = devicePixelRatio() * (height() - y - 1);

  QOpenGLFramebufferObjectFormat fbo_format;
  fbo_format.setSamples(0);
  QOpenGLFramebufferObject fbo(1, 1, fbo_format);

  glBindFramebuffer(GL_READ_FRAMEBUFFER, defaultFramebufferObject());
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo.handle());
  glBlitFramebuffer(scaled_x,
                    scaled_y,
                    scaled_x + 1,
                    scaled_y + 1,
                    0,
                    0,
                    1,
                    1,
                    GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT,
                    GL_NEAREST);

  fbo.bind();
  std::array<uint8_t, 3> color;
  glReadPixels(0, 0, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, color.data());
  fbo.release();

  const size_t index = RGBToIndex(color[0], color[1], color[2]);

  if (index < selection_buffer_.size()) {
    const char buffer_type = selection_buffer_[index].second;
    if (buffer_type == SELECTION_BUFFER_IMAGE_IDX) {
      selected_image_id_ = static_cast<image_t>(selection_buffer_[index].first);
      selected_point3D_id_ = kInvalidPoint3DId;
      ShowImageInfo(selected_image_id_);
    } else if (buffer_type == SELECTION_BUFFER_POINT_IDX) {
      selected_image_id_ = kInvalidImageId;
      selected_point3D_id_ = selection_buffer_[index].first;
      ShowPointInfo(selection_buffer_[index].first);
    } else {
      selected_image_id_ = kInvalidImageId;
      selected_point3D_id_ = kInvalidPoint3DId;
      image_viewer_widget_->hide();
    }
  } else {
    selected_image_id_ = kInvalidImageId;
    selected_point3D_id_ = kInvalidPoint3DId;
    image_viewer_widget_->hide();
  }

  // Re-enable, since temporarily disabled above.
  glEnable(GL_MULTISAMPLE);

  selection_buffer_.clear();

  UploadPointData();
  UploadImageData();
  UploadPointConnectionData();
  UploadImageConnectionData();
  UploadImageTimeData();

  update();
}

void ModelViewerWidget::SelectMoviewGrabberView(const size_t view_idx) {
  selected_movie_grabber_view_ = view_idx;
  UploadMovieGrabberData();
  update();
}

QImage ModelViewerWidget::GrabImage() {
  makeCurrent();

  DisableCoordinateGrid();

  paintGL();

  const int scaled_width = static_cast<int>(devicePixelRatio() * width());
  const int scaled_height = static_cast<int>(devicePixelRatio() * height());

  QOpenGLFramebufferObjectFormat fbo_format;
  fbo_format.setSamples(0);
  QOpenGLFramebufferObject fbo(scaled_width, scaled_height, fbo_format);

  glBindFramebuffer(GL_READ_FRAMEBUFFER, defaultFramebufferObject());
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo.handle());
  glBlitFramebuffer(0,
                    0,
                    scaled_width,
                    scaled_height,
                    0,
                    0,
                    scaled_width,
                    scaled_height,
                    GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT,
                    GL_NEAREST);

  fbo.bind();
  QImage image(scaled_width, scaled_height, QImage::Format_RGB888);
  glReadPixels(0,
               0,
               scaled_width,
               scaled_height,
               GL_RGB,
               GL_UNSIGNED_BYTE,
               image.bits());
  fbo.release();

  EnableCoordinateGrid();

  return image.mirrored();
}

void ModelViewerWidget::GrabMovie() { movie_grabber_widget_->show(); }

void ModelViewerWidget::ShowPointInfo(const point3D_t point3D_id) {
  point_viewer_widget_->Show(point3D_id);
}

void ModelViewerWidget::ShowImageInfo(const image_t image_id) {
  image_viewer_widget_->ShowImageWithId(image_id);
}

float ModelViewerWidget::PointSize() const { return point_size_; }

float ModelViewerWidget::ImageSize() const { return image_size_; }

void ModelViewerWidget::SetPointSize(const float point_size) {
  point_size_ = point_size;
}

void ModelViewerWidget::SetImageSize(const float image_size) {
  image_size_ = image_size;
  UploadImageData();
}

void ModelViewerWidget::SetBackgroundColor(const float r,
                                           const float g,
                                           const float b) {
  background_color_[0] = r;
  background_color_[1] = g;
  background_color_[2] = b;
  update();
}

void ModelViewerWidget::mousePressEvent(QMouseEvent* event) {
  if (mouse_press_timer_.isActive()) {  // Select objects (2. click)
    mouse_is_pressed_ = false;
    mouse_press_timer_.stop();
    selection_buffer_.clear();
    SelectObject(event->pos().x(), event->pos().y());
  } else {  // Set timer to remember 1. click
    mouse_press_timer_.setSingleShot(true);
    mouse_press_timer_.start(kDoubleClickInterval);
    mouse_is_pressed_ = true;
    prev_mouse_pos_ = event->pos();
  }
  event->accept();
}

void ModelViewerWidget::mouseReleaseEvent(QMouseEvent* event) {
  mouse_is_pressed_ = false;
  event->accept();
}

void ModelViewerWidget::mouseMoveEvent(QMouseEvent* event) {
  if (mouse_is_pressed_) {
    if (event->buttons() & Qt::RightButton ||
        (event->buttons() & Qt::LeftButton &&
         event->modifiers() & Qt::ControlModifier)) {
      TranslateView(event->pos().x(),
                    event->pos().y(),
                    prev_mouse_pos_.x(),
                    prev_mouse_pos_.y());
    } else if (event->buttons() & Qt::LeftButton) {
      RotateView(event->pos().x(),
                 event->pos().y(),
                 prev_mouse_pos_.x(),
                 prev_mouse_pos_.y());
    }
  }
  prev_mouse_pos_ = event->pos();
  event->accept();
}

void ModelViewerWidget::wheelEvent(QWheelEvent* event) {
  // We don't mind whether horizontal or vertical scroll.
  const float delta = event->angleDelta().x() + event->angleDelta().y();
  if (event->modifiers().testFlag(Qt::ControlModifier)) {
    ChangePointSize(delta);
  } else if (event->modifiers().testFlag(Qt::AltModifier)) {
    ChangeCameraSize(delta);
  } else if (event->modifiers().testFlag(Qt::ShiftModifier)) {
    ChangeNearPlane(delta);
  } else {
    ChangeFocusDistance(delta);
  }
  event->accept();
}

void ModelViewerWidget::SetupPainters() {
  makeCurrent();

  coordinate_axes_painter_.Setup();
  coordinate_grid_painter_.Setup();

  point_painter_.Setup();
  point_connection_painter_.Setup();

  image_line_painter_.Setup();
  image_triangle_painter_.Setup();
  image_connection_painter_.Setup();
  image_time_line_painter_.Setup();

  movie_grabber_path_painter_.Setup();
  movie_grabber_line_painter_.Setup();
  movie_grabber_triangle_painter_.Setup();
}

void ModelViewerWidget::SetupView() {
  point_size_ = kInitPointSize;
  image_size_ = kInitImageSize;
  focus_distance_ = kInitFocusDistance;
  model_view_matrix_.setToIdentity();
  model_view_matrix_.translate(0, 0, -focus_distance_);
  model_view_matrix_.rotate(225, 1, 0, 0);
  model_view_matrix_.rotate(-45, 0, 1, 0);
}

void ModelViewerWidget::Upload() {
  point_colormap_->Prepare(cameras, images, points3D, reg_image_ids);
  image_colormap_->Prepare(cameras, images, points3D, reg_image_ids);

  ComposeProjectionMatrix();

  UploadPointData();
  UploadImageData();
  UploadMovieGrabberData();
  UploadPointConnectionData();
  UploadImageConnectionData();
  UploadImageTimeData();

  update();
}

void ModelViewerWidget::UploadCoordinateGridData() {
  makeCurrent();

  const float scale = ZoomScale();

  // View center grid.
  std::vector<LinePainter::Data> grid_data(3);

  grid_data[0].point1 = PointPainter::Data(-20 * scale,
                                           0,
                                           0,
                                           kGridColor(0),
                                           kGridColor(1),
                                           kGridColor(2),
                                           kGridColor(3));
  grid_data[0].point2 = PointPainter::Data(20 * scale,
                                           0,
                                           0,
                                           kGridColor(0),
                                           kGridColor(1),
                                           kGridColor(2),
                                           kGridColor(3));

  grid_data[1].point1 = PointPainter::Data(0,
                                           -20 * scale,
                                           0,
                                           kGridColor(0),
                                           kGridColor(1),
                                           kGridColor(2),
                                           kGridColor(3));
  grid_data[1].point2 = PointPainter::Data(0,
                                           20 * scale,
                                           0,
                                           kGridColor(0),
                                           kGridColor(1),
                                           kGridColor(2),
                                           kGridColor(3));

  grid_data[2].point1 = PointPainter::Data(0,
                                           0,
                                           -20 * scale,
                                           kGridColor(0),
                                           kGridColor(1),
                                           kGridColor(2),
                                           kGridColor(3));
  grid_data[2].point2 = PointPainter::Data(0,
                                           0,
                                           20 * scale,
                                           kGridColor(0),
                                           kGridColor(1),
                                           kGridColor(2),
                                           kGridColor(3));

  coordinate_grid_painter_.Upload(grid_data);

  // Coordinate axes.
  std::vector<LinePainter::Data> axes_data(3);

  axes_data[0].point1 = PointPainter::Data(
      0, 0, 0, kXAxisColor(0), kXAxisColor(1), kXAxisColor(2), kXAxisColor(3));
  axes_data[0].point2 = PointPainter::Data(50 * scale,
                                           0,
                                           0,
                                           kXAxisColor(0),
                                           kXAxisColor(1),
                                           kXAxisColor(2),
                                           kXAxisColor(3));

  axes_data[1].point1 = PointPainter::Data(
      0, 0, 0, kYAxisColor(0), kYAxisColor(1), kYAxisColor(2), kYAxisColor(3));
  axes_data[1].point2 = PointPainter::Data(0,
                                           50 * scale,
                                           0,
                                           kYAxisColor(0),
                                           kYAxisColor(1),
                                           kYAxisColor(2),
                                           kYAxisColor(3));

  axes_data[2].point1 = PointPainter::Data(
      0, 0, 0, kZAxisColor(0), kZAxisColor(1), kZAxisColor(2), kZAxisColor(3));
  axes_data[2].point2 = PointPainter::Data(0,
                                           0,
                                           50 * scale,
                                           kZAxisColor(0),
                                           kZAxisColor(1),
                                           kZAxisColor(2),
                                           kZAxisColor(3));

  coordinate_axes_painter_.Upload(axes_data);
}

void ModelViewerWidget::UploadPointData(const bool selection_mode) {
  makeCurrent();

  std::vector<PointPainter::Data> data;

  // Assume we want to display the majority of points
  data.reserve(points3D.size());

  const size_t min_track_len =
      static_cast<size_t>(options_->render->min_track_len);

  if (selected_image_id_ == kInvalidImageId &&
      images.count(selected_image_id_) == 0) {
    for (const auto& point3D : points3D) {
      if (point3D.second.Error() <= options_->render->max_error &&
          point3D.second.Track().Length() >= min_track_len) {
        PointPainter::Data painter_point;

        painter_point.x = static_cast<float>(point3D.second.XYZ(0));
        painter_point.y = static_cast<float>(point3D.second.XYZ(1));
        painter_point.z = static_cast<float>(point3D.second.XYZ(2));

        Eigen::Vector4f color;
        if (selection_mode) {
          const size_t index = selection_buffer_.size();
          selection_buffer_.push_back(
              std::make_pair(point3D.first, SELECTION_BUFFER_POINT_IDX));
          color = IndexToRGB(index);

        } else if (point3D.first == selected_point3D_id_) {
          color = kSelectedPointColor;
        } else {
          color = point_colormap_->ComputeColor(point3D.first, point3D.second);
        }

        painter_point.r = color(0);
        painter_point.g = color(1);
        painter_point.b = color(2);
        painter_point.a = color(3);

        data.push_back(painter_point);
      }
    }
  } else {  // Image selected
    const auto& selected_image = images[selected_image_id_];
    for (const auto& point3D : points3D) {
      if (point3D.second.Error() <= options_->render->max_error &&
          point3D.second.Track().Length() >= min_track_len) {
        PointPainter::Data painter_point;

        painter_point.x = static_cast<float>(point3D.second.XYZ(0));
        painter_point.y = static_cast<float>(point3D.second.XYZ(1));
        painter_point.z = static_cast<float>(point3D.second.XYZ(2));

        Eigen::Vector4f color;
        if (selection_mode) {
          const size_t index = selection_buffer_.size();
          selection_buffer_.push_back(
              std::make_pair(point3D.first, SELECTION_BUFFER_POINT_IDX));
          color = IndexToRGB(index);
        } else if (selected_image.HasPoint3D(point3D.first)) {
          color = kSelectedImagePlaneColor;
        } else if (point3D.first == selected_point3D_id_) {
          color = kSelectedPointColor;
        } else {
          color = point_colormap_->ComputeColor(point3D.first, point3D.second);
        }

        painter_point.r = color(0);
        painter_point.g = color(1);
        painter_point.b = color(2);
        painter_point.a = color(3);

        data.push_back(painter_point);
      }
    }
  }

  point_painter_.Upload(data);
}

void ModelViewerWidget::UploadPointConnectionData() {
  makeCurrent();

  std::vector<LinePainter::Data> line_data;

  if (selected_point3D_id_ == kInvalidPoint3DId ||
      !options_->render->selected_image_connections) {
    // No point selected, or rendering not desired, so upload empty data
    point_connection_painter_.Upload(line_data);
    return;
  }

  const auto& point3D = points3D[selected_point3D_id_];

  // 3D point position.
  LinePainter::Data line;
  line.point1 = PointPainter::Data(static_cast<float>(point3D.XYZ(0)),
                                   static_cast<float>(point3D.XYZ(1)),
                                   static_cast<float>(point3D.XYZ(2)),
                                   kSelectedPointColor(0),
                                   kSelectedPointColor(1),
                                   kSelectedPointColor(2),
                                   0.8f);

  // All images in which 3D point is observed.
  for (const auto& track_el : point3D.Track().Elements()) {
    const Image& conn_image = images[track_el.image_id];
    const Eigen::Vector3f conn_proj_center =
        conn_image.ProjectionCenter().cast<float>();
    line.point2 = PointPainter::Data(conn_proj_center(0),
                                     conn_proj_center(1),
                                     conn_proj_center(2),
                                     kSelectedPointColor(0),
                                     kSelectedPointColor(1),
                                     kSelectedPointColor(2),
                                     0.8f);
    line_data.push_back(line);
  }

  point_connection_painter_.Upload(line_data);
}

void ModelViewerWidget::UploadImageData(const bool selection_mode) {
  makeCurrent();

  std::vector<LinePainter::Data> line_data;
  line_data.reserve(8 * reg_image_ids.size());

  std::vector<TrianglePainter::Data> triangle_data;
  triangle_data.reserve(2 * reg_image_ids.size());

  for (const image_t image_id : reg_image_ids) {
    const Image& image = images[image_id];
    const Camera& camera = cameras[image.CameraId()];

    Eigen::Vector4f plane_color;
    Eigen::Vector4f frame_color;
    if (selection_mode) {
      const size_t index = selection_buffer_.size();
      selection_buffer_.push_back(
          std::make_pair(image_id, SELECTION_BUFFER_IMAGE_IDX));
      plane_color = frame_color = IndexToRGB(index);
    } else {
      if (image_id == selected_image_id_) {
        plane_color = kSelectedImagePlaneColor;
        frame_color = kSelectedImageFrameColor;
      } else {
        image_colormap_->ComputeColor(image, &plane_color, &frame_color);
      }
    }

    // Lines are not colored with the indexed color in selection mode, so do not
    // show them, so they do not block the selection process
    if (frame_color(3) > 0.f) {
        BuildImageModel(image,
                        camera,
                        image_size_,
                        plane_color,
                        frame_color,
                        &triangle_data,
                        selection_mode ? nullptr : &line_data);
    }
  }

  image_line_painter_.Upload(line_data);
  image_triangle_painter_.Upload(triangle_data);
}

void ModelViewerWidget::UploadImageConnectionData() {
  makeCurrent();

  std::vector<LinePainter::Data> line_data;
  std::vector<image_t> image_ids;

  if (!options_->render->selected_image_connections &&
      !options_->render->image_connections) {
    image_connection_painter_.Upload(line_data);
    return;
  } else if (selected_image_id_ != kInvalidImageId) {
    // Show connections to selected images
    image_ids.push_back(selected_image_id_);
  } else if (options_->render->image_connections) {
    // Show all connections
    image_ids = reg_image_ids;
  } else {  // Disabled, so upload empty data
    image_connection_painter_.Upload(line_data);
    return;
  }

  for (const image_t image_id : image_ids) {
    const Image& image = images.at(image_id);

    const Eigen::Vector3f proj_center = image.ProjectionCenter().cast<float>();

    // Collect all connected images
    std::unordered_set<image_t> conn_image_ids;

    for (const Point2D& point2D : image.Points2D()) {
      if (point2D.HasPoint3D()) {
        const Point3D& point3D = points3D[point2D.point3D_id];
        for (const auto& track_elem : point3D.Track().Elements()) {
          conn_image_ids.insert(track_elem.image_id);
        }
      }
    }

    // Selected image in the center.
    LinePainter::Data line;
    line.point1 = PointPainter::Data(proj_center(0),
                                     proj_center(1),
                                     proj_center(2),
                                     kSelectedImageFrameColor(0),
                                     kSelectedImageFrameColor(1),
                                     kSelectedImageFrameColor(2),
                                     0.8f);

    // All connected images to the selected image.
    for (const image_t conn_image_id : conn_image_ids) {
      const Image& conn_image = images[conn_image_id];
      const Eigen::Vector3f conn_proj_center =
          conn_image.ProjectionCenter().cast<float>();
      line.point2 = PointPainter::Data(conn_proj_center(0),
                                       conn_proj_center(1),
                                       conn_proj_center(2),
                                       kSelectedImageFrameColor(0),
                                       kSelectedImageFrameColor(1),
                                       kSelectedImageFrameColor(2),
                                       0.8f);
      line_data.push_back(line);
    }
  }

  image_connection_painter_.Upload(line_data);
}

void ModelViewerWidget::UploadMovieGrabberData() {
  makeCurrent();

  std::vector<LinePainter::Data> path_data;
  path_data.reserve(movie_grabber_widget_->views.size());

  std::vector<LinePainter::Data> line_data;
  line_data.reserve(4 * movie_grabber_widget_->views.size());

  std::vector<TrianglePainter::Data> triangle_data;
  triangle_data.reserve(2 * movie_grabber_widget_->views.size());

  if (movie_grabber_widget_->views.size() > 0) {
    const Image& image0 = movie_grabber_widget_->views[0];
    Eigen::Vector3f prev_proj_center = image0.ProjectionCenter().cast<float>();

    for (size_t i = 1; i < movie_grabber_widget_->views.size(); ++i) {
      const Image& image = movie_grabber_widget_->views[i];
      const Eigen::Vector3f curr_proj_center =
          image.ProjectionCenter().cast<float>();
      LinePainter::Data path;
      path.point1 = PointPainter::Data(prev_proj_center(0),
                                       prev_proj_center(1),
                                       prev_proj_center(2),
                                       kSelectedImagePlaneColor(0),
                                       kSelectedImagePlaneColor(1),
                                       kSelectedImagePlaneColor(2),
                                       kSelectedImagePlaneColor(3));
      path.point2 = PointPainter::Data(curr_proj_center(0),
                                       curr_proj_center(1),
                                       curr_proj_center(2),
                                       kSelectedImagePlaneColor(0),
                                       kSelectedImagePlaneColor(1),
                                       kSelectedImagePlaneColor(2),
                                       kSelectedImagePlaneColor(3));
      path_data.push_back(path);
      prev_proj_center = curr_proj_center;
    }

    // Setup dummy camera with same settings as current OpenGL viewpoint.
    const float kDefaultImageWdith = 2048.0f;
    const float kDefaultImageHeight = 1536.0f;
    const float focal_length =
        -2.0f * std::tan(DegToRad(kFieldOfView) / 2.0f) * kDefaultImageWdith;
    Camera camera;
    camera.InitializeWithId(SimplePinholeCameraModel::model_id,
                            focal_length,
                            kDefaultImageWdith,
                            kDefaultImageHeight);

    // Build all camera models
    for (size_t i = 0; i < movie_grabber_widget_->views.size(); ++i) {
      const Image& image = movie_grabber_widget_->views[i];
      Eigen::Vector4f plane_color;
      Eigen::Vector4f frame_color;
      if (i == selected_movie_grabber_view_) {
        plane_color = kSelectedImagePlaneColor;
        frame_color = kSelectedImageFrameColor;
      } else {
        plane_color = kMovieGrabberImagePlaneColor;
        frame_color = kMovieGrabberImageFrameColor;
      }

      if (frame_color(3) > 0.f) {
        BuildImageModel(image,
                        camera,
                        image_size_,
                        plane_color,
                        frame_color,
                        &triangle_data,
                        &line_data);
      }
    }
  }

  movie_grabber_path_painter_.Upload(path_data);
  movie_grabber_line_painter_.Upload(line_data);
  movie_grabber_triangle_painter_.Upload(triangle_data);
}

void ModelViewerWidget::UploadImageTimeData(const bool selection_mode) {
  makeCurrent();

  const size_t num_images = reg_image_ids.size();
  if (num_images == 0) {
    return;
  }

  std::vector<LinePainter::Data> line_data;
  line_data.reserve(reg_image_ids.size());

  std::vector<double> pc_x, pc_y, pc_z;
  pc_x.reserve(num_images);
  pc_y.reserve(num_images);
  pc_z.reserve(num_images);
  for (size_t i = 0; i < num_images; ++i) {
    const auto& image_name = name_nums_to_image_indices[i];
    const size_t image_id_parsed = image_name.first;
    const image_t image_id = reg_image_ids[image_name.second];
    const Image& image = images[image_id];

    // TODO: Figure out if this center of projection is correct after changing it to not use the inverse projection matrix, post big change
    // https://github.com/colmap/colmap/commit/67029ad21205fac3d149e06000c1e20bf4be1b80#diff-dd10257411e1c7799c86cc92b2e6423b66eb2f1b2f2511a9c3f2ce46dc41b66f
    //
    // const Eigen::Matrix<float, 3, 4> inv_proj_matrix =
    //     image.InverseProjectionMatrix().cast<float>();
    // const Eigen::Vector3f projection_center = inv_proj_matrix.rightCols<1>();
    const auto projection_center = image.ProjectionCenter().cast<float>();
    pc_x.push_back(projection_center(0));
    pc_y.push_back(projection_center(1));
    pc_z.push_back(projection_center(2));
  }
  const int window_size = std::min(static_cast<int>(num_images), 6);
  const std::vector<size_t> inliers_x = RollingOutlierFilter(pc_x, window_size);
  const std::vector<size_t> inliers_y = RollingOutlierFilter(pc_y, window_size);
  const std::vector<size_t> inliers_z = RollingOutlierFilter(pc_z, window_size);
  std::vector<size_t> inliers_xy, inliers_xyz;
  std::set_intersection(inliers_x.begin(), inliers_x.end(), inliers_y.begin(),
                        inliers_y.end(), std::back_inserter(inliers_xy));
  std::set_intersection(inliers_xy.begin(), inliers_xy.end(), inliers_z.begin(),
                        inliers_z.end(), std::back_inserter(inliers_xyz));

  // Output the discarded outliers and two inlier values on either end.
  for (size_t i = 1; i < inliers_xyz.size(); ++i) {
    const auto first_outlier = inliers_xyz[i - 1] + 1;
    if (first_outlier != inliers_xyz[i]) {
      std::cout << StringPrintf("outlier indices: [%zu,%zu]\n", first_outlier,
                                inliers_xyz[i] - 1);
      for (size_t j = inliers_xyz[i - 1]; j <= inliers_xyz[i]; ++j) {
        const auto& file_num_and_index = name_nums_to_image_indices[j];
        const size_t file_num = file_num_and_index.first;
        const image_t image_id = reg_image_ids[file_num_and_index.second];
        const std::string lier =
            (j == inliers_xyz[i - 1] || j == inliers_xyz[i]) ? " inlier"
                                                             : "outlier";
        std::cout << StringPrintf(
            "%s env [%zu]: file=out%05zu.png, image_id=%zu, x=%.2f, "
            "y=%.2f, z=%.2f\n",
            lier.c_str(), j, file_num, image_id, pc_x[j], pc_y[j], pc_z[j]);
      }
    }
  }

  // // Useful for debugging when inliers is returning something crazy.
  // for (size_t i = 0; i < inliers_xyz.size() && i < 100; ++i) {
  //   const size_t j = inliers_xyz[i];
  //   if (j >= num_images) {
  //       std::cout << StringPrintf("%02d: !! %zu >= %zu\n", i, j, num_images);
  //   } else {
  //       std::cout << StringPrintf("%02d: %zu - x=%.4f, y=%.4f, z=%.4f\n", i,
  //       j, pc_x[j], pc_y[j], pc_z[j]);
  //   }
  // }

  const Eigen::Vector4f kpath_color = kSelectedFixedPathColor;
  Eigen::Vector4f frame_color, plane_color;
  for (size_t i = 1; i < inliers_xyz.size(); ++i) {
    size_t prev = inliers_xyz[i - 1];
    size_t curr = inliers_xyz[i];

    bool is_invis = false;
    const image_t image_id_0 =
        reg_image_ids[name_nums_to_image_indices[i - 1].second];
    const image_t image_id_1 =
        reg_image_ids[name_nums_to_image_indices[i].second];
    image_colormap_->ComputeColor(images[image_id_0], &plane_color,
                                  &frame_color);
    if (frame_color(3) < 0.01f) {
      is_invis = true;
    }
    image_colormap_->ComputeColor(images[image_id_1], &plane_color,
                                  &frame_color);
    if (frame_color(3) < 0.01f) {
      is_invis = true;
    }

    if (!is_invis) {
      line_data.emplace_back(
          PointPainter::Data(pc_x[prev], pc_y[prev], pc_z[prev], kpath_color(0),
                             kpath_color(1), kpath_color(2), kpath_color(3)),
          PointPainter::Data(pc_x[curr], pc_y[curr], pc_z[curr], kpath_color(0),
                             kpath_color(1), kpath_color(2), kpath_color(3)));
    }
  }

  image_time_line_painter_.Upload(line_data);
}

void ModelViewerWidget::ComposeProjectionMatrix() {
  projection_matrix_.setToIdentity();
  if (options_->render->projection_type ==
      RenderOptions::ProjectionType::PERSPECTIVE) {
    projection_matrix_.perspective(
        kFieldOfView, AspectRatio(), near_plane_, kFarPlane);
  } else if (options_->render->projection_type ==
             RenderOptions::ProjectionType::ORTHOGRAPHIC) {
    const float extent = OrthographicWindowExtent();
    projection_matrix_.ortho(-AspectRatio() * extent,
                             AspectRatio() * extent,
                             -extent,
                             extent,
                             near_plane_,
                             kFarPlane);
  }
}

float ModelViewerWidget::ZoomScale() const {
  // "Constant" scale factor w.r.t. zoom-level.
  return 2.0f * std::tan(static_cast<float>(DegToRad(kFieldOfView)) / 2.0f) *
         std::abs(focus_distance_) / height();
}

float ModelViewerWidget::AspectRatio() const {
  return static_cast<float>(width()) / static_cast<float>(height());
}

float ModelViewerWidget::OrthographicWindowExtent() const {
  return std::tan(DegToRad(kFieldOfView) / 2.0f) * focus_distance_;
}

Eigen::Vector3f ModelViewerWidget::PositionToArcballVector(
    const float x, const float y) const {
  Eigen::Vector3f vec(2.0f * x / width() - 1, 1 - 2.0f * y / height(), 0.0f);
  const float norm2 = vec.squaredNorm();
  if (norm2 <= 1.0f) {
    vec.z() = std::sqrt(1.0f - norm2);
  } else {
    vec = vec.normalized();
  }
  return vec;
}

}  // namespace colmap
