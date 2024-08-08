## fbgemm.dll

- IDE에서 가상 환경을 생성하고 환경 내에 pytorch를 설치한 뒤, import 했을 때 해당 *.dll*파일을 발견할 수 없다는 에러 발생

```
OSError: [WinError 126] 지정된 모듈을 찾을 수 없습니다. Error loading "c:\Users\dhsmf\anaconda3\envs\stt_env\lib\site-packages\torch\lib\fbgemm.dll" or one of its dependencies.
```

- *Visual Studio C/C++ Compiler* 를 설치하여 해결

---
[Ref.1](https://www.youtube.com/watch?v=sbQPGyVbePY) : Youtube 영상
[VS Compiler Site](https://visualstudio.microsoft.com/ko/vs/features/cplusplus/)
