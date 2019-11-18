﻿using Microsoft.AI.Skills.SkillInterfacePreview;
using Microsoft.Toolkit.Uwp.Helpers;
using Microsoft.Toolkit.Uwp.UI.Controls;
using SKU_FaceTransformation.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading;
using System.Threading.Tasks;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.Graphics.Imaging;
using Windows.Media;
using Windows.Storage;
using Windows.Storage.Pickers;
using Windows.Storage.Streams;
using Windows.UI;
using Windows.UI.Popups;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Media.Imaging;
using Windows.UI.Xaml.Navigation;
using Windows.UI.Xaml.Shapes;

// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace SKU_FaceTransformation
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        // Skill-related variables
        private FaceSentimentAnalyzerDescriptor m_skillDescriptor = null;
        private FaceSentimentAnalyzerSkill m_skill = null;
        private FaceSentimentAnalyzerBinding m_binding = null;

        // UI-related variables
        private SoftwareBitmapSource m_bitmapSource = new SoftwareBitmapSource(); // used to render an image from a file
        private FaceSentimentRenderer m_faceSentimentRenderer = null; // used to render a face rectangle on top of an iamge
        private IReadOnlyList<ISkillExecutionDevice> m_availableExecutionDevices = null;
        private uint m_cameraFrameWidth, m_cameraFrameHeight;
        private bool m_isCameraFrameDimensionInitialized = false;
        private enum FrameSourceToggledType { None, ImageFile, Camera };
        private FrameSourceToggledType m_currentFrameSourceToggled = FrameSourceToggledType.None;

        
        // Synchronization
        private SemaphoreSlim m_lock = new SemaphoreSlim(1);
        public MainPage()
        {
            this.InitializeComponent();
        }
        private async void Page_Loaded(object sender, RoutedEventArgs e)
        {
            // Initialize helper class used to render the skill results on screen
            m_faceSentimentRenderer = new FaceSentimentRenderer(UICanvasOverlay, UISentiment);

            try
            {
                // Instatiate skill descriptor to display details about the skill and populate UI
                m_skillDescriptor = new FaceSentimentAnalyzerDescriptor();
                m_availableExecutionDevices = await m_skillDescriptor.GetSupportedExecutionDevicesAsync();

                // Show skill description members in UI
                UISkillName.Text = m_skillDescriptor.Information.Name;

                UISkillDescription.Text = m_skillDescriptor.Information.Description;
                int featureIndex = 0;
                foreach (var featureDesc in m_skillDescriptor.InputFeatureDescriptors)
                {
                    UISkillInputDescription.Text += featureDesc.Description;
                    if (featureIndex++ < m_skillDescriptor.InputFeatureDescriptors.Count - 1)
                    {
                        UISkillInputDescription.Text += "\n----\n";
                    }
                }

                featureIndex = 0;
                foreach (var featureDesc in m_skillDescriptor.OutputFeatureDescriptors)
                {
                    UISkillOutputDescription.Text += featureDesc.Description;
                    if (featureIndex++ < m_skillDescriptor.OutputFeatureDescriptors.Count - 1)
                    {
                        UISkillOutputDescription.Text += "\n----\n";
                    }
                }

                if (m_availableExecutionDevices.Count == 0)
                {
                    UISkillOutputDetails.Text = "No execution devices available, this skill cannot run on this device";
                }
                else
                {
                    // Display available execution devices
                    UISkillExecutionDevices.ItemsSource = m_availableExecutionDevices.Select((device) => device.Name);
                    UISkillExecutionDevices.SelectedIndex = 0;

                    // Alow user to interact with the app
                    UIButtonFilePick.IsEnabled = true;
                    UICameraToggle.IsEnabled = true;
                    UIButtonFilePick.Focus(FocusState.Keyboard);
                }
            }
            catch (Exception ex)
            {
                await new MessageDialog(ex.Message).ShowAsync();
            }

            // Register callback for if camera preview encounters an issue
            UICameraPreview.PreviewFailed += UICameraPreview_PreviewFailed;
        }
        private async void UIButtonFilePick_Click(object sender, RoutedEventArgs e)
        {
            // Stop Camera preview
            UICameraPreview.Stop();
            if (UICameraPreview.CameraHelper != null)
            {
                await UICameraPreview.CameraHelper.CleanUpAsync();
            }
            UICameraPreview.Visibility = Visibility.Collapsed;
            UIImageViewer.Visibility = Visibility.Visible;

            // Disable subsequent trigger of this event callback 
            UICameraToggle.IsEnabled = false;
            UIButtonFilePick.IsEnabled = false;

            await m_lock.WaitAsync();

            try
            {
                // Initialize skill with the selected supported device
                m_skill = await m_skillDescriptor.CreateSkillAsync(m_availableExecutionDevices[UISkillExecutionDevices.SelectedIndex]) as FaceSentimentAnalyzerSkill;

                // Instantiate a binding object that will hold the skill's input and output resource
                m_binding = await m_skill.CreateSkillBindingAsync() as FaceSentimentAnalyzerBinding;

                var frame = await LoadVideoFrameFromFilePickedAsync();
                if (frame != null)
                {
                    await m_bitmapSource.SetBitmapAsync(frame.SoftwareBitmap);
                    UIImageViewer.Source = m_bitmapSource;

                    UIImageViewer_SizeChanged(null, null);

                    await RunSkillAsync(frame);
                }

                m_skill = null;
                m_binding = null;

                m_currentFrameSourceToggled = FrameSourceToggledType.ImageFile;
            }
            catch (Exception ex)
            {
                await (new MessageDialog(ex.Message)).ShowAsync();
                m_currentFrameSourceToggled = FrameSourceToggledType.None;
            }

            m_lock.Release();

            // Enable subsequent trigger of this event callback
            UIButtonFilePick.IsEnabled = true;
            UICameraToggle.IsEnabled = true;
        }
        public static IAsyncOperation<VideoFrame> LoadVideoFrameFromFilePickedAsync()
        {
            return AsyncInfo.Run(async (token) =>
            {
                // Trigger file picker to select an image file
                FileOpenPicker fileOpenPicker = new FileOpenPicker();
                fileOpenPicker.SuggestedStartLocation = PickerLocationId.PicturesLibrary;
                fileOpenPicker.FileTypeFilter.Add(".jpg");
                fileOpenPicker.FileTypeFilter.Add(".png");
                fileOpenPicker.ViewMode = PickerViewMode.Thumbnail;
                StorageFile selectedStorageFile = await fileOpenPicker.PickSingleFileAsync();

                if (selectedStorageFile == null)
                {
                    return null;
                }

                VideoFrame resultFrame = null;
                SoftwareBitmap softwareBitmap = null;
                using (IRandomAccessStream stream = await selectedStorageFile.OpenAsync(FileAccessMode.Read))
                {
                    // Create the decoder from the stream 
                    BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream);

                    // Get the SoftwareBitmap representation of the file in BGRA8 format
                    softwareBitmap = await decoder.GetSoftwareBitmapAsync();

                    // Convert to friendly format for UI display purpose
                    softwareBitmap = SoftwareBitmap.Convert(softwareBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
                }

                // Encapsulate the image in a VideoFrame instance
                resultFrame = VideoFrame.CreateWithSoftwareBitmap(softwareBitmap);

                return resultFrame;
            });
        }
        private async void UICameraToggle_Click(object sender, RoutedEventArgs e)
        {
            await m_lock.WaitAsync();
            try
            {
                UICameraPreview.Stop();
                if (UICameraPreview.CameraHelper != null)
                {
                    await UICameraPreview.CameraHelper.CleanUpAsync();
                }
                m_isCameraFrameDimensionInitialized = false;

                // Initialize skill with the selected supported device
                m_skill = await m_skillDescriptor.CreateSkillAsync(m_availableExecutionDevices[UISkillExecutionDevices.SelectedIndex]) as FaceSentimentAnalyzerSkill;

                // Instantiate a binding object that will hold the skill's input and output resource
                m_binding = await m_skill.CreateSkillBindingAsync() as FaceSentimentAnalyzerBinding;

                // Initialize the CameraPreview control, register frame arrived event callback
                UIImageViewer.Visibility = Visibility.Collapsed;
                UICameraPreview.Visibility = Visibility.Visible;
                await UICameraPreview.StartAsync();

                UICameraPreview.CameraHelper.FrameArrived += CameraHelper_FrameArrived;
                m_currentFrameSourceToggled = FrameSourceToggledType.Camera;
            }
            catch (Exception ex)
            {
                await (new MessageDialog(ex.Message)).ShowAsync();
                m_currentFrameSourceToggled = FrameSourceToggledType.None;
            }
            finally
            {
                m_lock.Release();
            }
        }
        private async void CameraHelper_FrameArrived(object sender, FrameEventArgs e)
        {
            try
            {
                // Use a lock to process frames one at a time and bypass processing if busy
                if (m_lock.Wait(0))
                {
                    uint cameraFrameWidth = UICameraPreview.CameraHelper.PreviewFrameSource.CurrentFormat.VideoFormat.Width;
                    uint cameraFrameHeight = UICameraPreview.CameraHelper.PreviewFrameSource.CurrentFormat.VideoFormat.Height;

                    // Allign overlay canvas and camera preview so that face detection rectangle looks right
                    if (!m_isCameraFrameDimensionInitialized || cameraFrameWidth != m_cameraFrameWidth || cameraFrameHeight != m_cameraFrameHeight)
                    {
                        m_cameraFrameWidth = UICameraPreview.CameraHelper.PreviewFrameSource.CurrentFormat.VideoFormat.Width;
                        m_cameraFrameHeight = UICameraPreview.CameraHelper.PreviewFrameSource.CurrentFormat.VideoFormat.Height;

                        await Dispatcher.RunAsync(Windows.UI.Core.CoreDispatcherPriority.Normal, () =>
                        {
                            UIImageViewer_SizeChanged(null, null);
                        });

                        m_isCameraFrameDimensionInitialized = true;
                    }

                    // Run the skill against the frame
                    await RunSkillAsync(e.VideoFrame);
                    m_lock.Release();
                }
                e.VideoFrame.Dispose();
            }
            catch (Exception ex)
            {
                // Show the error
                await Dispatcher.RunAsync(Windows.UI.Core.CoreDispatcherPriority.Normal, () => UISkillOutputDetails.Text = ex.Message);
                m_lock.Release();
            }
        }
        private async void UICameraPreview_PreviewFailed(object sender, PreviewFailedEventArgs e)
        {
            await new MessageDialog(e.Error).ShowAsync();
        }
        private async Task RunSkillAsync(VideoFrame frame)
        {
            // Update input image and run the skill against it
            await m_binding.SetInputImageAsync(frame);
            await m_skill.EvaluateAsync(m_binding);

            await Dispatcher.RunAsync(Windows.UI.Core.CoreDispatcherPriority.Normal, () =>
            {
                // Retrieve result
                if (!m_binding.IsFaceFound)
                {
                    // if no face found, hide the rectangle in the UI
                    m_faceSentimentRenderer.IsVisible = false;
                    UISkillOutputDetails.Text = "No face found";
                }
                else // Display the face rectangle and sentiment in the UI
                {
                    m_faceSentimentRenderer.Update(m_binding.FaceRectangle, m_binding.PredominantSentiment);
                    m_faceSentimentRenderer.IsVisible = true;
                    var scores = (m_binding["FaceSentimentScores"].FeatureValue as SkillFeatureTensorFloatValue).GetAsVectorView();
                    UISkillOutputDetails.Text = "";
                    for (int i = 0; i < (int)SentimentType.contempt; i++)
                    {
                        UISkillOutputDetails.Text += $"{(SentimentType)i} : {scores[i]} {(i == (int)m_binding.PredominantSentiment ? " <<------" : "")} \n";
                    }
                }
            });
        }
        private void UISkillExecutionDevices_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            switch (m_currentFrameSourceToggled)
            {
                case FrameSourceToggledType.ImageFile:
                    UIButtonFilePick_Click(null, null);
                    break;
                case FrameSourceToggledType.Camera:
                    UICameraToggle_Click(null, null);
                    break;
                default:
                    break;
            }
        }
        private void UIImageViewer_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            if (UIImageViewer.Visibility == Visibility.Visible) // we are using an image file that we stretch, match UI control dimension
            {
                UICanvasOverlay.Width = UIImageViewer.ActualWidth;
                UICanvasOverlay.Height = UIImageViewer.ActualHeight;
            }
            else // we are using a camera preview, make sure the aspect ratio is honored when rendering the face rectangle
            {
                float aspectRatio = (float)m_cameraFrameWidth / m_cameraFrameHeight;
                UICanvasOverlay.Width = aspectRatio >= 1.0f ? UICameraPreview.ActualWidth : UICameraPreview.ActualHeight * aspectRatio;
                UICanvasOverlay.Height = aspectRatio >= 1.0f ? UICameraPreview.ActualWidth / aspectRatio : UICameraPreview.ActualHeight;
            }
        }
    }
    internal class FaceSentimentRenderer
    {
        private Canvas m_canvas;
        private TextBlock m_sentimentControl;
        private Rectangle m_rectangle = new Rectangle();
        private Dictionary<SentimentType, string> m_emojis = new Dictionary<SentimentType, string>
        {
            { SentimentType.neutral, "😒" },
            { SentimentType.happiness, "😄" },
            { SentimentType.surprise, "😲" },
            { SentimentType.sadness, "😢" },
            { SentimentType.anger, "😡" },
            { SentimentType.disgust, "😝" },
            { SentimentType.fear, "😱" },
            { SentimentType.contempt, "😤" }
        };

        /// <summary>
        /// FaceSentimentRenderer constructor
        /// </summary>
        /// <param name="canvas"></param>
        public FaceSentimentRenderer(Canvas canvas, TextBlock sentimentControl)
        {
            m_canvas = canvas;
            m_sentimentControl = sentimentControl;
            m_rectangle = new Rectangle() { Stroke = new SolidColorBrush(Colors.Red), StrokeThickness = 2 };
            m_canvas.Children.Add(m_rectangle);
            IsVisible = false;
        }

        /// <summary>
        /// Set visibility of FaceSentimentRendere UI controls
        /// </summary>
        public bool IsVisible
        {
            get
            {
                return m_canvas.Visibility == Visibility.Visible;
            }
            set
            {
                m_canvas.Visibility = value ? Visibility.Visible : Visibility.Collapsed;
                m_sentimentControl.Visibility = value ? Visibility.Visible : Visibility.Collapsed;
            }
        }

        /// <summary>
        /// Update coordinates of face rectangle and predominant sentiment passsed as parameter
        /// </summary>
        /// <param name="coordinates"></param>
        public void Update(IReadOnlyList<float> coordinates, SentimentType sentiment)
        {
            if (coordinates == null)
            {
                return;
            }
            if (coordinates.Count != 4)
            {
                throw new Exception("you can only pass a set of 4 float coordinates (left, top, right, bottom) to this method");
            }
            m_rectangle.Width = (coordinates[2] - coordinates[0]) * m_canvas.Width;
            m_rectangle.Height = (coordinates[3] - coordinates[1]) * m_canvas.Height;
            Canvas.SetLeft(m_rectangle, coordinates[0] * m_canvas.Width);
            Canvas.SetTop(m_rectangle, coordinates[1] * m_canvas.Height);

            m_sentimentControl.Text = $"{m_emojis[sentiment]}";
        }
    }
}
