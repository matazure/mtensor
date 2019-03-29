pipeline{

	stages {
		stage('linux') {
			stages {
				stage('x86-64'){
					agent { docker { image 'matazure/ci4tensor:gcc-ubuntu18.04'  }  }
					environment {
						CXX = g++
						CC = gcc
					}
					steps {
						sh './script/build.sh'
					}
				}
			}
		}
		stage('windows') {
			
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
