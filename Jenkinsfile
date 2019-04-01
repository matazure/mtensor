pipeline{
	agent none
	stages {
		stage('Matazure Tensor CI'){
			parallel {
				
				stage('linux-x64') {
					agent { 
						docker { 
							image 'matazure/ci4tensor:gcc-ubuntu18.04'  
						}
					}
					environment {
						CXX = 'g++'
						CC = 'gcc'
					}
					stages {
						stage('build') {
							steps {
								sh './script/build.sh'
							}
						}
					}
				}
				
				stage('linux-x64-cuda') {
					agent {
						docker {
							image 'matazure/ci4tensor:cuda10.1-ubuntu18.04'
							args '--runtime=nvidia'
						}
					}
					environment {
						CXX = 'g++-6'
						CC = 'gcc-6'
					}
					stages {
						stage('build') {
							steps {
								sh 'mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON && cmake --build .'
							}
						}
						stage('test') {
							steps {
								sh 'echo hicudatest'
							}
						}
					}
				}
				
				stage('android-armv7') {
					stages {	
						stage('build') {
							agent {
								docker {
									image 'matazure/ci4tensor:linaro-armv7'
								}
							}
							steps {
								sh './script/build_android.sh'
							}
						}
						stage {
							stages {
								agent {
									lable 'rpi-armv7'
								}
								stage ('test'){
									steps {
										sh 'echo test'
									}
								}
								stage ('benchmark'){
									steps {
										sh 'echo benchmark'
									}
								}
							}
						}
					}
				}
				
				stage('windows-x64') {
					agent {
						label 'win10-x64'
					}
					stages {
						stage('build'){
							steps {
								bat 'call ./script/build_win.bat'
							}
						}
					}
				}
				
				stage('android-armv7') {
					agent {
						docker {
							image 'matazure/ci4tensor:android-ndk-r16b'
						}
					}
					stages {	
						stage('build') {
							steps {
								sh './script/build_android.sh'
							}
						}
						stage('test') {
							steps {
								sh 'echo armv7-test'
							}
						}	
					}
				}
				
				stage('macos-x64') {
					agent {
						label 'macos-x64'
					}
					stages {
						stage('build') {
							steps {
								sh './script/build.sh'
							}
						}
					}
				}
			}
		}
	}
	post {
		always {
			echo 'This will always run'
		}
		success {
			echo 'This will run only ifsuccessful'
		}
		failure {
			echo 'This will run only iffailed'
		}
		unstable {
			echo 'This will run only if th run was marked as unstable'
		}
		changed {
			echo 'This will run only if th state of the Pipeline has changed'
			echo 'For example, if thePipeline was previously failing bu is now successful'
		}
	}
}
