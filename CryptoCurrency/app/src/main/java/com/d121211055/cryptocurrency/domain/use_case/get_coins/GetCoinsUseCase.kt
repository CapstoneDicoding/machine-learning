package com.d121211055.cryptocurrency.domain.use_case.get_coins

import android.net.http.HttpException
import android.os.Build
import android.os.ext.SdkExtensions
import com.d121211055.cryptocurrency.common.Resource
import com.d121211055.cryptocurrency.data.remote.dto.toCoin
import com.d121211055.cryptocurrency.domain.model.Coin
import com.d121211055.cryptocurrency.domain.repository.CoinRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.io.IOException
import javax.inject.Inject

class GetCoinsUseCase @Inject constructor(
    private val repository: CoinRepository
) {
    operator fun invoke(): Flow<Resource<List<Coin>>> = flow {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && SdkExtensions.getExtensionVersion(
                Build.VERSION_CODES.S) >= 7) {
            try {
                emit(Resource.Loading())
                val coins = repository.getCoins().map { it.toCoin() }
                emit(Resource.Success(coins))
            } catch (e:HttpException){
                emit(Resource.Error(e.localizedMessage ?: "An unexpected error occured"))
            } catch (e:IOException){
                emit(Resource.Error("Couldn't reach server. Check your internet connection"))
            }
        }
    }
}