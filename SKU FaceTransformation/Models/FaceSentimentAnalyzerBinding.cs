
using Microsoft.AI.Skills.SkillInterface;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Windows.AI.MachineLearning;
using Windows.Foundation;
using Windows.Media;

namespace SKU_FaceTransformation.Models
{
    /// Defines the set of possible emotion label scored by this skill
    public enum SentimentType
    {
        neutral = 0,
        happiness,
        surprise,
        sadness,
        anger,
        disgust,
        fear,
        contempt
    };
    public sealed class FaceSentimentAnalyzerBinding : IReadOnlyDictionary<string, ISkillFeature>, ISkillBinding
    {
        private VisionSkillBindingHelper m_bindingHelper = null;
        // WinML related member variables
        internal LearningModelBinding m_winmlBinding = null;
        // ISkillBinding
        public ISkillExecutionDevice Device => m_bindingHelper.Device;

        // IReadOnlyDictionary
        public bool ContainsKey(string key) => m_bindingHelper.ContainsKey(key);
        public bool TryGetValue(string key, out ISkillFeature value) => m_bindingHelper.TryGetValue(key, out value);
        public ISkillFeature this[string key] => m_bindingHelper[key];
        public IEnumerable<string> Keys => m_bindingHelper.Keys;
        public IEnumerable<ISkillFeature> Values => m_bindingHelper.Values;
        public int Count => m_bindingHelper.Count;
        public IEnumerator<KeyValuePair<string, ISkillFeature>> GetEnumerator() => m_bindingHelper.AsEnumerable().GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => m_bindingHelper.AsEnumerable().GetEnumerator();

        // Constructor
        internal FaceSentimentAnalyzerBinding(
            ISkillDescriptor descriptor,
            ISkillExecutionDevice device,
            LearningModelSession session)
        {
            m_bindingHelper = new VisionSkillBindingHelper(descriptor, device);

            // Create WinML binding
            m_winmlBinding = new LearningModelBinding(session);
        }
        public IAsyncAction SetInputImageAsync(VideoFrame frame)
        {

            return m_bindingHelper.SetInputImageAsync(frame);

        }
        /// Returns whether or not a face is found given the bound outputs
        public bool IsFaceFound
        {
            get
            {
                ISkillFeature feature = null;
                if (m_bindingHelper.TryGetValue(FaceSentimentAnalyzerConst.SKILL_OUTPUTNAME_FACERECTANGLE, out feature))
                {
                    var faceRect = (feature.FeatureValue as SkillFeatureTensorFloatValue).GetAsVectorView();
                    return !(faceRect[0] == 0.0f &&
                        faceRect[1] == 0.0f &&
                        faceRect[2] == 0.0f &&
                        faceRect[3] == 0.0f);
                }
                else
                {
                    return false;
                }
            }
        }

        /// Returns the sentiment with the highest score
        public SentimentType PredominantSentiment
        {
            get
            {
                SentimentType predominantSentiment = SentimentType.neutral;
                ISkillFeature feature = null;
                if (m_bindingHelper.TryGetValue(FaceSentimentAnalyzerConst.SKILL_OUTPUTNAME_FACESENTIMENTSCORES, out feature))
                {
                    var faceSentimentScores = (feature.FeatureValue as SkillFeatureTensorFloatValue).GetAsVectorView();

                    float maxScore = float.MinValue;
                    for (int i = 0; i < faceSentimentScores.Count; i++)
                    {
                        if (faceSentimentScores[i] > maxScore)
                        {
                            predominantSentiment = (SentimentType)i;
                            maxScore = faceSentimentScores[i];
                        }
                    }
                }

                return predominantSentiment;
            }
        }

        /// Returns the face rectangle
        public IReadOnlyList<float> FaceRectangle
        {
            get
            {
                ISkillFeature feature = null;
                if (m_bindingHelper.TryGetValue(FaceSentimentAnalyzerConst.SKILL_OUTPUTNAME_FACERECTANGLE, out feature))
                {
                    return (feature.FeatureValue as SkillFeatureTensorFloatValue).GetAsVectorView();
                }
                else
                {
                    return null;
                }
            }
        }
    }
}
