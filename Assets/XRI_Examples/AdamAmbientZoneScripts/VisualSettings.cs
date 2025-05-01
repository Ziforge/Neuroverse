using UnityEngine;
using UnityEngine.Audio;
using UnityEngine.UI;

public class VisualSettings : MonoBehaviour
{
    [SerializeField] Slider visSlider;

    DarknessOverlayGradient darkOverlayG;
    private void Start() {
        //SetVolume(PlayerPrefs.GetFloat("SavedMasterVolume, 0.0")); 
        darkOverlayG = GameObject.FindGameObjectWithTag("ND_Sound").GetComponent<DarknessOverlayGradient>();
        SetVal(0);
    }

    public void SetVal(float _value) { 
        if(_value < 1) { //* needed?
            _value = .001f;
        }

        RefreshSlider(_value);
        //darkness slider here based off of -->* masterMixer.SetFloat("MasterVolume", Mathf.Log10(_value / 100) * 20f);   
        //darkOverlayG.SetZoneDarkness(_value / 100f);
    }

    public void SetValFromSlider() {
        SetVal(visSlider.value);
    }

    public void RefreshSlider(float _value) {
        visSlider.value = _value;
    }

    public void QuarterSlider() {
        SetVal(25);
    }

}