// Azure deployment: Azure OpenAI (AI Foundry) + Container Registry + Container Apps.
// The API authenticates to Azure OpenAI with a Managed Identity - no keys stored anywhere.
//
//   az group create -n rg-knowledge-copilot -l swedencentral
//   az deployment group create -g rg-knowledge-copilot -f infra/azure/main.bicep -p image=<acr>.azurecr.io/knowledge-copilot:latest

@description('Base name used for all resources')
param name string = 'kcopilot'
param location string = resourceGroup().location
@description('Full image reference, e.g. myacr.azurecr.io/knowledge-copilot:latest')
param image string
param chatModel string = 'gpt-4.1-mini'
param chatModelVersion string = '2025-04-14'
param embeddingModel string = 'text-embedding-3-small'

var suffix = uniqueString(resourceGroup().id)

// ---- Azure OpenAI ----------------------------------------------------------
resource openai 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: '${name}-aoai-${suffix}'
  location: location
  kind: 'OpenAI'
  sku: { name: 'S0' }
  properties: {
    customSubDomainName: '${name}-aoai-${suffix}'
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: true // keyless only
  }
}

resource chatDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openai
  name: chatModel
  sku: { name: 'GlobalStandard', capacity: 50 }
  properties: {
    model: { format: 'OpenAI', name: chatModel, version: chatModelVersion }
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
  }
}

resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openai
  name: embeddingModel
  sku: { name: 'Standard', capacity: 120 }
  properties: { model: { format: 'OpenAI', name: embeddingModel, version: '1' } }
  dependsOn: [chatDeployment] // deployments on one account must be created sequentially
}

// ---- Container infrastructure ------------------------------------------------
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: '${name}acr${suffix}'
  location: location
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false }
}

resource logs 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${name}-logs-${suffix}'
  location: location
  properties: { sku: { name: 'PerGB2018' }, retentionInDays: 30 }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${name}-env-${suffix}'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${name}-id-${suffix}'
  location: location
}

// Cognitive Services OpenAI User
resource openaiUserRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(openai.id, identity.id, 'openai-user')
  scope: openai
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
  }
}

// AcrPull
resource acrPullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, identity.id, 'acrpull')
  scope: acr
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
  }
}

// Qdrant as a sidecar-less internal container app with a persistent volume would need Azure Files;
// for a first deployment we run Qdrant in embedded mode inside the API container (QDRANT_URL=./qdrant_data).
resource api 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${name}-api'
  location: location
  identity: { type: 'UserAssigned', userAssignedIdentities: { '${identity.id}': {} } }
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: { external: true, targetPort: 8000, transport: 'http' }
      registries: [{ server: acr.properties.loginServer, identity: identity.id }]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: image
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: [
            { name: 'LLM_PROVIDER', value: 'azure' }
            { name: 'EMBEDDING_PROVIDER', value: 'azure' }
            { name: 'AZURE_OPENAI_ENDPOINT', value: openai.properties.endpoint }
            { name: 'AZURE_CHAT_DEPLOYMENT', value: chatDeployment.name }
            { name: 'AZURE_EMBEDDING_DEPLOYMENT', value: embeddingDeployment.name }
            { name: 'AZURE_CLIENT_ID', value: identity.properties.clientId }
            { name: 'QDRANT_URL', value: './qdrant_data' }
          ]
          probes: [
            { type: 'Liveness', httpGet: { path: '/health', port: 8000 }, periodSeconds: 30 }
            { type: 'Readiness', httpGet: { path: '/health', port: 8000 }, initialDelaySeconds: 10 }
          ]
        }
      ]
      scale: { minReplicas: 1, maxReplicas: 3, rules: [{ name: 'http', http: { metadata: { concurrentRequests: '20' } } }] }
    }
  }
  dependsOn: [acrPullRole, openaiUserRole]
}

output apiUrl string = 'https://${api.properties.configuration.ingress.fqdn}'
output openaiEndpoint string = openai.properties.endpoint
output acrLoginServer string = acr.properties.loginServer
