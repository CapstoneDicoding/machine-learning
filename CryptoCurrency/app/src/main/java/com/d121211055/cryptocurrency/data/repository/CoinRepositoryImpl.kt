package com.d121211055.cryptocurrency.data.repository

import com.d121211055.cryptocurrency.data.remote.CoinPaprikaApi
import com.d121211055.cryptocurrency.data.remote.dto.CoinDetailDto
import com.d121211055.cryptocurrency.data.remote.dto.CoinDto
import com.d121211055.cryptocurrency.domain.repository.CoinRepository
import javax.inject.Inject

class CoinRepositoryImpl @Inject constructor(
    private val api: CoinPaprikaApi
): CoinRepository{
    override suspend fun getCoins(): List<CoinDto> {
        return api.getCoins()
    }

    override suspend fun getCoinById(coinId: String): CoinDetailDto {
        return api.getCoinById(coinId)
    }
}