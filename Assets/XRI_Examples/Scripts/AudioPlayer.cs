using UnityEngine;
using UnityEngine.Audio;
using UnityEngine.UI;

public class AudioPlayer : MonoBehaviour
{
    public Toggle playStopButton;
    public Button nextButton;
    public Button previousButton;

    public AudioSource[] Songs;

    public GameObject playIcon;
    public GameObject pauseIcon;

    private int currentSongIndex = 0;

    void Start()
    {
        playStopButton.onValueChanged.AddListener(playOrPause);
        nextButton.onClick.AddListener(nextSong);
        previousButton.onClick.AddListener(previousSong);

        playOrPause(playStopButton.isOn);
        nextSong();
        previousSong();

        StopAllSongs();
        Songs[currentSongIndex].Play();
        playStopButton.isOn = false; // Playing
        playOrPause(false);
    }

    // Update is called once per frame
    void playOrPause(bool _pause)
    {
        if (_pause)
        {
            Songs[currentSongIndex].Pause();
            playIcon.SetActive(true);
            pauseIcon.SetActive(false);
            Debug.Log("UnityDebug playOrPause: " + "Pause");
        }
        else
        {
            Songs[currentSongIndex].Play();
            playIcon.SetActive(false);
            pauseIcon.SetActive(true);
            Debug.Log("UnityDebug playOrPause: " + "Play");
        }
    }
    void nextSong()
    {
        Songs[currentSongIndex].Stop();
        currentSongIndex = (currentSongIndex + 1) % Songs.Length;
        Songs[currentSongIndex].Play();
        playStopButton.isOn = false; // Ensure play state is correct
        playOrPause(false);
    }
    void previousSong()
    {
        Songs[currentSongIndex].Stop();
        currentSongIndex = (currentSongIndex - 1 + Songs.Length) % Songs.Length;
        Songs[currentSongIndex].Play();
        playStopButton.isOn = false;
        playOrPause(false);
    }

    void StopAllSongs()
    {
        foreach (var song in Songs)
        {
            song.Stop();
        }
    }
}
