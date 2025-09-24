pipeline {
    agent any
    environment {
        scannerHome = tool 'SonarQube Scanner'
    }
    stages {
        stage('SonarQube analysis') {
            steps {
                withCredentials([string(credentialsId: 'Jenkins_Sonarqube', variable: 'SONARQUBE_TOKEN')]) {
                    withSonarQubeEnv('SonarQube Scanner') {
                        sh """
                        ${scannerHome}/bin/sonar-scanner -X \
                          -Dsonar.projectKey=customerstaffinteraction \
                          -Dsonar.projectName=customerstaffinteraction \
                          -Dsonar.sources=. \
                          -Dsonar.language=py \
                          -Dsonar.sourceEncoding=UTF-8 \
                          -Dsonar.python.version=3 \
                          -Dsonar.exclusions="**/modules/**,**/libraries/**" \
                          -Dsonar.token=$SONARQUBE_TOKEN \
                          -Dsonar.host.url=http://sonarqube.proeffico.com/
                        """
                    }
                }
            }

        }
        stage("Quality Gate") {
            steps {
                timeout(time: 1, unit: 'HOURS') {
                    waitForQualityGate abortPipeline: true
                }
            }
        }

        stage('Conda Environment Setup') {
            steps {
                sshPublisher(publishers: [sshPublisherDesc(
                    configName: 'pythongpu',
                    transfers: [sshTransfer(
                        cleanRemote: false,
                        execCommand: '''
                            sudo bash -c "
                                cd /var/www/html/customerstaffinteraction
                                source /root/miniconda/etc/profile.d/conda.sh
                                conda activate /root/miniconda/envs/csi/
                                echo '→ Active conda environment:'
                                conda info --envs
                                echo '→ Which Python is being used:'
                                which python
                                echo '→ Python version:'
                                python --version
                                echo '→ Running deploy.sh'
                                bash -x deploy.sh
                            "
                        ''',
                        execTimeout: 600000,
                        flatten: false,
                        makeEmptyDirs: false,
                        noDefaultExcludes: false,
                        patternSeparator: '[, ]+',
                        remoteDirectory: '',
                        removePrefix: '',
                        sourceFiles: ''
                    )],
                    usePromotionTimestamp: false,
                    useWorkspaceInPromotion: false,
                    verbose: true
                )])
            }
        }
    }
    
    post {
        always {
            cleanWs(patterns: [[pattern: 'dependency-check-report.xml', type: 'EXCLUDE']])
        }
    }
}